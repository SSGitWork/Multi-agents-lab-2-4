# Lab 2.4 — Cross-Framework Comparison

**Module 2 · Lab 2.4**
Build Autonomous Multi-Agent Systems · Saras AI Institute

---

## What you are building

In Labs 2.1–2.3 you built the PM + Coder pipeline entirely in LangGraph.
In this lab you step back and ask: was LangGraph the right tool? How
does the same PM planning step look and perform when expressed in AutoGen
and CrewAI?

You will replicate the PM agent's two-step planning workflow (spec
generation → task decomposition) in both AutoGen and CrewAI, measure
the token cost and API call count of each implementation, and write a
structured comparison document that justifies a framework recommendation
for the capstone project.

This is the only lab in Module 2 with a written deliverable alongside code.

---

## Repo structure

```
lab2.4/
├── .devcontainer/
│   └── devcontainer.json
├── implementations/
│   ├── langgraph_impl.py       # Pre-provided reference — do not modify
│   ├── autogen_impl.py         # ← YOUR WORK (Task 1)
│   └── crewai_impl.py          # ← YOUR WORK (Task 2)
├── tests/
│   └── test_lab24.py           # Test suite — do not modify
├── llm_client.py               # Shared LLM config — do not modify
├── shared_types.py             # PlanningResult schema — do not modify
├── run_comparison.py           # Comparison runner — do not modify
├── COMPARISON.md               # ← YOUR WORK (Task 3)
└── requirements.txt
```

**You edit three files: `autogen_impl.py`, `crewai_impl.py`, and `COMPARISON.md`.**

---

## Setup

```bash
python -c "from llm_client import get_llm_client; print('Client OK')"
python -c "import autogen; print('AutoGen OK')"
python -c "from crewai import Agent; print('CrewAI OK')"
```

---

## Your tasks

### Task 1 — `implementations/autogen_impl.py`

Implement `run(requirement) -> PlanningResult` using AutoGen.

Use two sequential single-turn conversations — one to produce the tech
spec and one to decompose it into tasks. The system prompts are the same
as the LangGraph reference. Return a `PlanningResult` with real token
counts read from the AutoGen chat history.

Read the full docstring in `autogen_impl.py` before starting — it
includes the exact AutoGen API calls, token counting approach, and
recommended `UserProxyAgent` settings.

### Task 2 — `implementations/crewai_impl.py`

Implement `run(requirement) -> PlanningResult` using CrewAI.

Use a sequential Crew with two agents (spec writer + task decomposer)
and two tasks with `context=[write_spec]` to pass the spec to the
decomposer. Return a `PlanningResult` with token counts from
`result.token_usage`.

Read the full docstring in `crewai_impl.py` before starting.

### Task 3 — `COMPARISON.md`

Fill in all six required sections of `COMPARISON.md` using the measured
data from `python run_comparison.py`. Every section marked **[REQUIRED]**
must be completed — the test suite checks for placeholder text and empty
table cells.

---

## Running the tests

```bash
# Schema and structural tests — no API calls required
python -m pytest tests/test_lab24.py -v -k "not live"

# Live API tests — makes real API calls (~$0.10 per full run)
python -m pytest tests/test_lab24.py -v live
```

Run the live tests once per implementation to verify your `PlanningResult`
schema is correct before filling in `COMPARISON.md`.

---

## Running the comparison

```bash
# Run all three frameworks and print the comparison table
python run_comparison.py

# Run a single framework to debug it in isolation
python run_comparison.py autogen
python run_comparison.py crewai
python run_comparison.py langgraph
```

The runner saves results to `comparison_results.json`. Use these figures
in `COMPARISON.md`.

---

## Success criteria

- `pytest tests/test_lab24.py -v -k "not live"` passes all non-live tests
- `pytest tests/test_lab24.py -v -m live` passes all live tests for both
  AutoGen and CrewAI
- `python run_comparison.py` prints a three-row comparison table with
  non-zero token counts for all three frameworks
- `COMPARISON.md` has all six required sections filled in with real data,
  no placeholder text remaining, and a specific recommendation backed by
  at least two measured data points

---

## Key concepts this lab teaches

**Framework abstraction cost.** AutoGen and CrewAI add convenience at the
cost of tokens. The LangGraph/raw-API implementation gives you exact control
over what goes into every prompt; framework implementations inject overhead
(AutoGen footers, CrewAI role/goal/backstory concatenation) that you can
measure but not fully suppress.

**The PlanningResult schema as a comparison harness.** By requiring all
three implementations to return the same TypedDict, you can compare them
objectively. The schema is the test fixture — it makes framework differences
measurable rather than impressionistic.

**Token counting is framework-specific.** LangGraph: `response.usage`.
AutoGen: buried in `chat_messages` dicts. CrewAI: `result.token_usage`.
None of them is hard, but none is the same. This matters for cost tracking
in production (Module 4).

**Reading a framework's source to understand overhead.** The most common
observation students make is that CrewAI's prompt is larger than expected
because the role/goal/backstory are concatenated. Looking at what actually
reaches the LLM (via Helicone) is more informative than reading the docs.

---

## Common errors

| Symptom | Likely cause |
|---------|-------------|
| AutoGen blocks waiting for keyboard input | `human_input_mode` not set to `"NEVER"` |
| AutoGen makes 4 calls instead of 2 | Missing `is_termination_msg=lambda _: True` |
| `total_tokens == 0` in AutoGen | `msg.get("usage")` returns None — try printing a chat message to see its structure |
| CrewAI JSON parse fails | `context=[write_spec]` injects extra prose around the JSON — apply fence stripping |
| COMPARISON.md tests fail | Placeholder text still present — search for `_(describe` in the file |
