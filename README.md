# ADK 2.0 Workflow Orchestration Lab

A hands-on lab in Google ADK 2.0's `Workflow` graph API — not a tutorial retread, but a working
implementation of every core orchestration pattern (parallel fan-out, deterministic routing, runtime-sized
fan-out, recursive spawning) with each level verified against real execution traces, not just "it ran
without an error." Several of the findings below only surfaced because the reference docs and the actually
installed package disagreed — the fix, each time, was to check the running code, not the docs.

**Trigger:** a YouTube video on graph-based agent orchestration (fan-out/JoinNode/router patterns) exposed
a gap — I assumed LangGraph was the only real option for explicit graph orchestration, not realizing ADK
2.0 added a first-class `Workflow` API covering the same ground. This project follows
[Google's ADK 2.0 orchestration codelab](https://colab.research.google.com/github/cuppibla/adk2-tutorial/blob/main/notebooks/adk2_orchestration_workshop.ipynb)
as a syllabus, adapted level-by-level into a real running agent instead of a Colab notebook.

## What was built, level by level

| Level | Pattern | Mechanism | Verified result |
|---|---|---|---|
| L0 | Single agent + tool | `Agent` + a deterministic Python tool (`pace_splits`) | The tool computed exact pace math correctly — but the model still restated one value off by a second when summarizing it in prose. Calling the right tool and reporting its result verbatim are two different guarantees. |
| L1 | First graph | `Workflow`: function node → `LlmAgent` node | The function node's event carried no `usageMetadata` (0 LLM cost) while feeding real data into the reasoning step downstream. |
| L2a | Parallel fan-out + join | `JoinNode` bundling 3 concurrent function nodes | Real wall-clock proof from event timestamps: 3 branches (1.0s / 1.5s / 2.0s simulated latency) finished in **~2.03s total**, not ~4.5s — parallel fan-out costs the slowest branch, not the sum. |
| L2b | Deterministic routing | Plain `if`/`elif` node emitting `Event(route=...)`, dict-edge to 3 specialist agents | Exactly **1 LLM call** ran (the matched branch) out of 3 possible specialists — 0 wasted calls, confirmed by trace. |
| L3a/b | Collaborative sub-agents | `sub_agents=[...]`, `mode="chat" \| "task" \| "single_turn"` | Already implemented in other projects — skipped here rather than re-derived. |
| L4a | Runtime-sized fan-out | `@node(parallel_worker=True)` over a decomposer's variable-length output | The LLM chose **6** sub-questions on one run (not the schema's minimum of 3) — width genuinely decided at runtime, not fixed in code. |
| L4b | Recursive spawning | `ctx.run_node()` calling the same node recursively, bounded by `MAX_DEPTH` | `MAX_DEPTH=2` correctly bounded a real run (14 total LLM calls) — but only provably so after reconstructing the true recursion depth from the nested output data, because the dev UI's debug path strings turned out to be misleading (see findings). |

## Findings

**A public ADK 2.0 Workflow reference documented the wrong syntax for conditional routing.**
The cheatsheet used while building L2b showed conditional routing as a 3-element tuple —
`(router_node, target_node, "ROUTE")`. Running it against the actual installed `google-adk` package threw
a 21-error `pydantic.ValidationError`. The real, working form for this install is a **dict-edge**:
`(router_node, {"ROUTE_A": target_a, "ROUTE_B": target_b})` — which is what an unrelated raw source (the
codelab notebook itself) had used all along. Lesson: a library's own pydantic model, inspected directly, beats a cheatsheet.

**That same error also proved something useful: an N-element tuple in `edges` is a literal chain.**
`(START, a, b, c)` and `[("START", a), (a, b), (b, c)]` are equivalent — every position in the tuple is
independently validated as "NodeLike."

**`ctx.run_node()` on a `parallel_worker=True` node always returns its result wrapped in a list** — even for
a single dynamic or recursive call with a plain dict input, not a list. Confirmed with an isolated,
no-LLM repro script before trusting it inside L4b's real recursive pipeline (`research_topic` calling
itself). Missing this would have silently produced a list-of-lists instead of a flat list of findings.

**The dev UI's `nodeInfo.path` debug strings are not reliable evidence of recursion depth** once a session
gets reused across multiple distinct workflow runs that share a `Workflow.name` — a real L4b run's debug
path showed what looked like 6 nested recursion levels, which would have meant `MAX_DEPTH=2` had failed.
Reconstructing the actual depth from the nested `children` arrays in the real output data showed the guard
had worked correctly the whole time (max depth reached: 2, exactly as configured). The debug string was
a red herring, not the ground truth.

**A parameter absent from a cheatsheet isn't necessarily absent from the library.**
`LlmAgent.sub_agents`, `LlmAgent.mode` (`Literal['chat', 'task', 'single_turn']`), and
`LlmAgent.input_schema` are all real, working pydantic fields on the installed `google-adk` — confirmed via
direct `model_fields` introspection — even though one reference document implied only some of them were
supported. Cuts both ways: don't trust an undocumented parameter blindly, but don't reject one either
without checking the actual installed model.

## Project structure

```
app/
  agent.py          # current state: L4b (runtime-sized decompose → recursive research fan-out → synthesize)
  fast_api_app.py    # FastAPI wrapper (scaffold-generated, unmodified)
tests/                # scaffold-generated test skeleton (unit/integration/eval), not the focus of this lab
```

`app/agent.py` was fully rewritten at each level rather than kept as separate files — the table above is
the record of what each version looked like and what it proved; the single git commit in this repo
captures only the final (L4b) state, not the incremental journey.

## Stack

- **Framework:** Google ADK 2.0 (`google-adk`) — `Workflow` graph API (`JoinNode`, `@node`, `ctx.run_node`)
  and the classic `LlmAgent` / `sub_agents` collaboration API
- **Model:** Gemini (`gemini-3.6-flash`) via Vertex AI
- **Tooling:** `agents-cli` scaffold, `uv`, the local ADK dev UI (`agents-cli playground`)

## Quick start

```bash
agents-cli install     # uv sync
agents-cli playground  # local dev UI at http://127.0.0.1:8080/dev-ui/?app=app, auto-reloads on save
```

## Related projects

- [account-health-ml-service](https://github.com/fabricioespel-bit/account-health-ml-service) — the
  project this one ran alongside, started as a parallel track while that project's Azure Synapse workspace
  was blocked on a support ticket.
