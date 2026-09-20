"""
Lab 2.4 — CrewAI Implementation
=================================

Replicate the PM agent's planning step using CrewAI.

CrewAI is a role-based multi-agent framework. Agents are defined with
explicit roles, goals, and backstories. Work is expressed as Tasks with
descriptions and expected outputs. A Crew orchestrates which agents run
which tasks in which order.

────────────────────────────────────────────────────────────────────────
CREWAI BACKGROUND
────────────────────────────────────────────────────────────────────────

The three main CrewAI primitives:

  Agent   — defines a specialist with role, goal, backstory, and an llm.
            The role + goal + backstory together act as the system prompt.

  Task    — a unit of work: a description of what to do, the agent that
            does it, and the expected_output (a description of the desired
            result format, not a schema).

  Crew    — assembles agents and tasks, sets process type (sequential or
            hierarchical), and executes via crew.kickoff().

A minimal single-agent, single-task crew:

    from crewai import Agent, Task, Crew, Process

    analyst = Agent(
        role="Data Analyst",
        goal="Summarise the data clearly",
        backstory="You are an expert at turning raw data into insights.",
        llm=llm_config,
        verbose=False,
    )
    task = Task(
        description="Summarise this data: {data}",
        expected_output="A bullet-point summary of the key findings.",
        agent=analyst,
    )
    crew = Crew(agents=[analyst], tasks=[task], process=Process.sequential)
    result = crew.kickoff(inputs={"data": "..."})
    output_text = result.raw   # the final string output

────────────────────────────────────────────────────────────────────────
YOUR TASK — implement `run(requirement)`
────────────────────────────────────────────────────────────────────────

Produce a PlanningResult using CrewAI. The result must have the same
tech_spec and tasks schema as the LangGraph implementation.

Recommended approach: one crew, two sequential tasks
-----------------------------------------------------
CrewAI's sequential process is a natural fit for the two-step workflow:

  Agent 1 — spec_writer
      role:      "Software Architect"
      goal:      "Produce a structured technical specification from a
                  user requirement"
      backstory: "You are a senior architect who writes clear, unambiguous
                  specs that developers can implement without follow-up."

  Task 1 — write_spec
      description:     "Given this requirement: {requirement}\n\nWrite a
                        technical specification with these five sections:
                        ## Overview, ## Functional Requirements,
                        ## Non-Functional Requirements, ## File Structure,
                        ## Constraints & Assumptions"
      expected_output: "A markdown technical specification with exactly
                        five ## sections, under 400 words, no TBDs."
      agent:           spec_writer

  Agent 2 — task_decomposer
      role:      "Technical Project Manager"
      goal:      "Decompose a technical specification into coding tasks"
      backstory: "You break down specs into concrete, ordered tasks that
                  developers can implement independently."

  Task 2 — decompose_tasks
      description:     "Decompose this specification into 2–4 coding tasks.
                        Return ONLY a JSON object with this schema: ..."
                        (include the full schema from LangGraph's prompt)
      expected_output: "A raw JSON object with a 'tasks' key containing
                        2–4 task dicts. No markdown fences."
      agent:           task_decomposer
      context:         [write_spec]   ← passes task 1's output as context

  Crew: Process.sequential, agents=[spec_writer, task_decomposer],
        tasks=[write_spec, decompose_tasks]

Extracting outputs
------------------
After crew.kickoff(inputs={"requirement": requirement}):

  result.raw          — the final task's output as a string
  result.tasks_output — list of TaskOutput objects, one per task

  To get task 1's output (the spec):
      spec_output = result.tasks_output[0].raw

  To get task 2's output (the JSON):
      tasks_json = result.tasks_output[1].raw

Token counting
--------------
CrewAI exposes token usage on the kickoff result:

    result.token_usage   — a UsageMetrics object with:
        prompt_tokens, completion_tokens, total_tokens

Use these directly. If they are zero (some versions), fall back to
recording 0 and explaining in the `notes` field.

JSON parsing
------------
Apply the same fence-stripping as the other implementations:

    text = text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()

notes field
-----------
Record at least two concrete observations about the CrewAI developer
experience compared to LangGraph. For example:
  - How does expressing roles/goals/backstory differ from a system_message?
  - How did the context=[write_spec] dependency feel compared to explicit
    variable passing?
  - Was token_usage easy to read?
  - Did the sequential process add overhead you could measure?

Signature
---------
def run(requirement: str) -> PlanningResult
"""

import json
from dotenv import load_dotenv
load_dotenv()

from crewai import Agent, Task, Crew, Process
from llm_client import get_crewai_llm
from shared_types import PlanningResult, TaskSpec
# llm = get_crewai_llm()


def _strip_fences(text: str) -> str:
    return text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()

def run(requirement: str) -> PlanningResult:
    """
    Run the PM planning step using CrewAI.

    Parameters
    ----------
    requirement : str
        The raw user requirement.

    Returns
    -------
    PlanningResult
        Spec, tasks, token usage, and implementation notes.
    """
    llm = get_crewai_llm()

    spec_writer = Agent(
        role="Software Architect",
        goal="Produce a structured technical specification from a user requirement",
        backstory=(
            "You are a senior architect who writes clear, unambiguous specs that developers can implement without follow-up."
        ),
        llm=llm,
        verbose=False,
    )
    task_decomposer = Agent(
        role="Technical Project Manager",
        goal="Decompose a technical specification into coding tasks",
        backstory=(
            "You break down specs into concrete, ordered tasks that developers can implement independently."
        ),
        llm=llm,
        verbose=False,
    )

    write_spec = Task(
        description=(
            "Given this requirement: {requirement}\n\nWrite a technical specification with these five sections:\n"
            "## Overview, ## Functional Requirements, ## Non-Functional Requirements, ## File Structure, ## Constraints & Assumptions"
        ),
        expected_output=(
            "A markdown technical specification with exactly five ## sections, under 400 words, no TBDs."
        ),
        agent=spec_writer,
    )
    decompose_tasks = Task(
        description=(
            "Decompose this specification into 2–4 coding tasks. Return ONLY a JSON object with this schema: "
            '{"tasks": [{"task_id": "task_1", "title": "<short imperative title>", "description": "<full self-contained spec>", "acceptance_criteria": ["<check>"], "status": "pending", "file_path": "<e.g. src/app.py>"}]}'
        ),
        expected_output="A raw JSON object with a 'tasks' key containing 2–4 task dicts. No markdown fences.",
        agent=task_decomposer,
        context=[write_spec],
    )

    crew = Crew(
        agents=[spec_writer, task_decomposer],
        tasks=[write_spec, decompose_tasks],
        process=Process.sequential,
    )
    result = crew.kickoff(inputs={"requirement": requirement})

    spec_text = result.tasks_output[0].raw
    tasks_text = result.tasks_output[1].raw
    usage = getattr(result, "token_usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or (prompt_tokens + completion_tokens))
    tasks = json.loads(_strip_fences(tasks_text))["tasks"]

    return {
        "framework": "crewai",
        "tech_spec": spec_text,
        "tasks": tasks,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "llm_calls": 2,
        "notes": (
            "CrewAI made the two-step workflow easy to express with sequential tasks and context passing. "
            "Token usage was available directly on the kickoff result, which was simpler than AutoGen's history-based extraction."
        ),
    }


if __name__ == "__main__":
    from shared_types import BENCHMARK_REQUIREMENT
    result = run(BENCHMARK_REQUIREMENT)
    print(f"[CrewAI] {result['llm_calls']} calls | "
          f"{result['total_tokens']} tokens | "
          f"{len(result['tasks'])} tasks")
    if result["notes"]:
        print(f"Notes: {result['notes']}")
