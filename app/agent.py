# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.events.event import Event
from google.adk.models import Gemini
from google.adk.workflow import RetryConfig, Workflow, node
from google.genai import types
from pydantic import BaseModel, Field


MODEL = "gemini-3.6-flash"

class DecomposerOutput(BaseModel):
    plan_summary: str = Field(
        description="One sentence explaining how you decomposed the question."
    )
    sub_questions: list[str] = Field(
        description=(
            "3-7 specific, non-overlapping sub-questions that together "
            "comprehensively answer the user's main question."
        ),
        min_length=3,
        max_length=7,
    )

class ResearchFinding(BaseModel):
    summary: str = Field(
        description="Concise 2-3 sentence summary of findings on this sub-question."
    )
    key_facts: list[str] = Field(
        description="3-5 specific factual points or actionable insights discovered.",
        min_length=2,
        max_length=5,
    )
    needs_deeper: bool = Field(
        description=(
            "True if these findings reveal a topic that warrants deeper "
            "recursive investigation. False if the question is fully answered."
        )
    )
    deeper_questions: list[str] = Field(
        default_factory=list,
        description=(
            "If needs_deeper is True, 1-3 specific, well-formed deeper "
            "questions to investigate. Empty list if needs_deeper is False."
        ),
        max_length=3,
    )

class DeepResearchBriefing(BaseModel):
    summary: str = Field(
        description="2-3 sentence overview synthesizing all the findings."
    )
    key_takeaways: list[str] = Field(
        description="3-6 actionable takeaways drawn from the findings.",
        min_length=3,
        max_length=6,
    )

def _model() -> Gemini:
    return Gemini(model=MODEL, retry_options=types.HttpRetryOptions(attempts=3))


def _coerce(data, model: type[BaseModel]) -> BaseModel:
    return data if isinstance(data, model) else model(**data)


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    parts = getattr(content, "parts", None) or []
    return "".join(getattr(p, "text", "") or "" for p in parts)

decompose_agent = LlmAgent(
    name="decompose_agent",
    model=_model(),
    output_schema=DecomposerOutput,
    instruction=(
        "You are a research coordinator for marathon/endurance questions. "
        "Break the user's open-ended question into 3-7 specific, "
        "non-overlapping, independently-researchable sub-questions that "
        "together comprehensively answer it."
    ),
)

research_agent = LlmAgent(
    name="research_agent",
    model=_model(),
    output_schema=ResearchFinding,
    instruction=(
        "You are a marathon research specialist. Given ONE specific "
        "research question, produce a finding: a 2-3 sentence summary and "
        "3-5 specific insights that directly answer it. Then decide: does "
        "this finding reveal a sub-topic that genuinely warrants deeper "
        "investigation (set needs_deeper=True and list 1-3 specific "
        "deeper_questions), or is the question fully answered as-is (set "
        "needs_deeper=False and leave deeper_questions empty)? Be "
        "conservative — only request deeper research when there's a "
        "genuinely complex sub-topic left unresolved."
    ),
)

synthesize_agent = LlmAgent(
    name="synthesize_agent",
    model=_model(),
    output_schema=DeepResearchBriefing,
    instruction=(
        "You are a marathon coach synthesizing a JSON list of research "
        "findings into one coherent briefing for the runner. Merge "
        "overlapping points, keep it concrete, cite specifics from the "
        "findings."
    ),
)

@node(rerun_on_resume=True)
async def decompose(ctx, node_input):
    """Runtime-sized fan-out: the LLM itself decides how many sub-questions (3-7)."""
    user_query = _extract_text(node_input)
    plan = _coerce(
        await ctx.run_node(decompose_agent, node_input=user_query),
        DecomposerOutput,
    )
    yield Event(
        output=[
            {"question": q, "original_query": user_query}
            for q in plan.sub_questions
        ]
    )

RESEARCH_RETRY = RetryConfig(max_attempts=3, initial_delay=2.0, backoff_factor=2.0)

MAX_DEPTH = 2

@node(parallel_worker=True, rerun_on_resume=True, retry_config=RESEARCH_RETRY)
async def research_topic(ctx, node_input):
    """One worker per sub-question; recursively spawns deeper children up to MAX_DEPTH."""
    question = node_input["question"]
    original_query = node_input.get("original_query", "")
    depth = node_input.get("depth", 0)

    finding = _coerce(
        await ctx.run_node(
            research_agent,
            node_input=f"ORIGINAL QUERY: {original_query}\n\nRESEARCH QUESTION: {question}",
        ),
        ResearchFinding,
    )

    children = []
    if finding.needs_deeper and depth < MAX_DEPTH:
        for deeper_question in finding.deeper_questions:
            child = await ctx.run_node(
                research_topic,
                node_input={
                    "question": deeper_question,
                    "original_query": original_query,
                    "depth": depth + 1,
                },
            )
            children.append(child[0] if isinstance(child, list) else child)

    yield Event(
        output={
            "question": question,
            "summary": finding.summary,
            "key_facts": finding.key_facts,
            "children": children,
        }
    )

@node(rerun_on_resume=True)
async def synthesize(ctx, node_input):
    briefing = _coerce(
        await ctx.run_node(
            synthesize_agent, node_input=json.dumps(node_input, indent=2)
        ),
        DeepResearchBriefing,
    )
    yield Event(output={"briefing": briefing.model_dump(), "findings": node_input})

root_agent = Workflow(
    name="l4a_flat_research",
    description="Decompose (runtime width) → flat parallel research → synthesize.",
    edges=[
        ("START", decompose),
        (decompose, research_topic),
        (research_topic, synthesize),
    ],
)

app = App(
    root_agent=root_agent,
    name="app",
)