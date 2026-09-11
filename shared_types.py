"""
Shared output schema for all three framework implementations.

All three implementations — LangGraph, AutoGen, and CrewAI — must
return a PlanningResult. This is what makes the comparison fair: the
same input, the same expected output shape, measured across three
different framework implementations.

You do not need to modify this file.
"""

from typing import TypedDict


class TaskSpec(TypedDict):
    """
    A single coding task as decomposed by the PM planner.
    Identical to CodingTask from Labs 2.1–2.3.
    """
    task_id:             str
    title:               str
    description:         str
    acceptance_criteria: list[str]
    status:              str
    file_path:           str


class PlanningResult(TypedDict):
    """
    The output every framework implementation must produce.

    Fields
    ------
    framework : str
        One of "langgraph", "autogen", "crewai".

    tech_spec : str
        The structured technical specification produced from the
        raw requirement.

    tasks : list[TaskSpec]
        The decomposed task list (2–4 tasks).

    prompt_tokens : int
        Total prompt tokens consumed across all LLM calls in this run.
        Read from the API response's usage object.

    completion_tokens : int
        Total completion tokens consumed across all LLM calls.

    total_tokens : int
        prompt_tokens + completion_tokens.

    llm_calls : int
        Number of LLM API calls made. The LangGraph reference
        implementation makes exactly 2 (one for spec, one for tasks).
        Record what your implementation actually makes.

    notes : str
        Any implementation notes, surprises, or framework quirks
        you observed while building this. Used in the comparison doc.
        Can be empty string if none.
    """
    framework:         str
    tech_spec:         str
    tasks:             list[TaskSpec]
    prompt_tokens:     int
    completion_tokens: int
    total_tokens:      int
    llm_calls:         int
    notes:             str


# The requirement all three implementations run against.
# Keep this constant — changing it invalidates the comparison.
BENCHMARK_REQUIREMENT = (
    "Build a small Flask REST API with two endpoints: "
    "GET /health that returns {\"status\": \"ok\", \"version\": \"1.0\"}, "
    "and GET /items that returns a hardcoded JSON list of five item objects, "
    "each with an 'id' (integer) and a 'name' (string) field. "
    "The app should run on port 8080."
)
