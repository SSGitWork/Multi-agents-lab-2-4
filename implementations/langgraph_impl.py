"""
LangGraph Implementation — pre-provided reference.

This is the PM agent's planning step from Lab 2.1, adapted to return a
PlanningResult so it can be compared against the AutoGen and CrewAI
implementations you will write.

You do not need to modify this file. Read it carefully — it is the
baseline your two implementations must match in output schema and
should be compared against in your written analysis.

Key things to notice
--------------------
- Two sequential LLM calls: one for the spec, one for task decomposition.
- Token counts are read directly from response.usage.
- The implementation is ~40 lines of logic (excluding prompts).
- State is managed as plain Python variables — no framework overhead.
"""

import json
from dotenv import load_dotenv
load_dotenv()

from llm_client import get_llm_client, MODEL
from shared_types import PlanningResult, TaskSpec

_SPEC_SYSTEM = """\
You are a senior software architect and Product Manager.
Given a natural language requirement, produce a structured technical
specification that a developer can act on without asking follow-up questions.

Write the spec in Markdown with exactly these five sections:
## Overview
## Functional Requirements
## Non-Functional Requirements
## File Structure
## Constraints & Assumptions

Rules:
- Be concrete and unambiguous. No "TBD", no open questions.
- List every file that will need to be created under File Structure.
- Keep the spec under 400 words.
"""

_DECOMPOSE_SYSTEM = """\
You are a technical project manager. Given a technical specification,
decompose it into 2–4 coding tasks for a developer to implement.

Return ONLY a valid JSON object — no markdown fences, no explanation.

Schema:
{
  "tasks": [
    {
      "task_id": "task_1",
      "title": "<short imperative title>",
      "description": "<full self-contained spec>",
      "acceptance_criteria": ["<check>"],
      "status": "pending",
      "file_path": "<e.g. src/app.py>"
    }
  ]
}

Rules:
- Between 2 and 4 tasks only.
- Order tasks by dependency (foundations first).
- Each description must be fully self-contained.
- Every task must have a unique file_path.
- status must be exactly "pending".
- Return ONLY the JSON object.
"""


def run(requirement: str) -> PlanningResult:
    """
    Run the LangGraph/plain-OpenAI PM planning step.

    Parameters
    ----------
    requirement : str
        The raw user requirement.

    Returns
    -------
    PlanningResult
        Spec, tasks, and token usage metrics.
    """
    client = get_llm_client()
    total_prompt = 0
    total_completion = 0
    calls = 0

    # Call 1 — build tech spec
    resp1 = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": _SPEC_SYSTEM},
            {"role": "user",   "content": f"Requirement: {requirement}"},
        ],
        temperature=0,
    )
    tech_spec = resp1.choices[0].message.content
    total_prompt     += resp1.usage.prompt_tokens
    total_completion += resp1.usage.completion_tokens
    calls += 1

    # Call 2 — decompose into tasks
    resp2 = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": _DECOMPOSE_SYSTEM},
            {"role": "user",   "content": f"Technical specification:\n\n{tech_spec}"},
        ],
        temperature=0,
    )
    text = resp2.choices[0].message.content.strip()
    text = text.lstrip("```json").lstrip("```").rstrip("```").strip()
    tasks: list[TaskSpec] = json.loads(text)["tasks"]
    total_prompt     += resp2.usage.prompt_tokens
    total_completion += resp2.usage.completion_tokens
    calls += 1

    return {
        "framework":         "langgraph",
        "tech_spec":         tech_spec,
        "tasks":             tasks,
        "prompt_tokens":     total_prompt,
        "completion_tokens": total_completion,
        "total_tokens":      total_prompt + total_completion,
        "llm_calls":         calls,
        "notes":             "",
    }


if __name__ == "__main__":
    from shared_types import BENCHMARK_REQUIREMENT
    result = run(BENCHMARK_REQUIREMENT)
    print(f"[LangGraph] {result['llm_calls']} calls | "
          f"{result['total_tokens']} tokens | "
          f"{len(result['tasks'])} tasks")
