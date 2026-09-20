"""
Lab 2.4 — AutoGen Implementation
==================================

Replicate the PM agent's planning step using Microsoft AutoGen.

AutoGen is a conversational multi-agent framework. Rather than explicit
sequential LLM calls, AutoGen agents converse with each other to produce
a result. Your task is to express the same two-step planning workflow
(spec → task decomposition) using AutoGen's agent abstractions, then
measure what the framework costs in tokens and API calls compared to the
LangGraph baseline.

────────────────────────────────────────────────────────────────────────
AUTOGEN BACKGROUND
────────────────────────────────────────────────────────────────────────

The two main AutoGen agent types:

  AssistantAgent   — an LLM-backed agent that generates responses.
                     Configured with a system_message that defines its
                     role and behaviour.

  UserProxyAgent   — acts as the "user" in a conversation. In this lab
                     it initiates the conversation and receives the final
                     response. Set human_input_mode="NEVER" so it runs
                     autonomously without waiting for keyboard input.

A minimal two-agent conversation:

    import autogen

    assistant = autogen.AssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant.",
        llm_config=llm_config,
    )
    user_proxy = autogen.UserProxyAgent(
        name="user",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        is_termination_msg=lambda msg: True,   # terminate after one reply
    )
    user_proxy.initiate_chat(assistant, message="Hello")
    last_msg = assistant.last_message()["content"]

────────────────────────────────────────────────────────────────────────
YOUR TASK — implement `run(requirement)`
────────────────────────────────────────────────────────────────────────

Produce a PlanningResult using AutoGen. The result must have the same
tech_spec and tasks schema as the LangGraph implementation. Token counts
must be real — read from AutoGen's chat history or usage tracking.

Recommended approach: two sequential conversations
--------------------------------------------------
Mirror the LangGraph implementation's two-call structure using two
separate AutoGen conversations:

  Conversation 1 — Spec writer
      Create an AssistantAgent with a system_message asking it to
      produce a structured technical specification (same five sections
      as the LangGraph implementation).
      Use a UserProxyAgent to initiate the conversation with the
      requirement, then extract the spec from the assistant's reply.

  Conversation 2 — Task decomposer
      Create a second AssistantAgent with a system_message asking it
      to decompose the spec into a JSON task list (same schema as
      LangGraph).
      Use a fresh UserProxyAgent to initiate the conversation with the
      spec, then parse the JSON from the assistant's reply.

Token counting
--------------
AutoGen does not expose usage objects as directly as the raw OpenAI API.
Two approaches:

  Option A — Wrap the OpenAI client with a counting proxy:
      Use the Helicone dashboard (all calls are already routed there).
      Read token counts from the response objects in the chat history:

      for msg in assistant.chat_messages[user_proxy]:
          # msg is a dict; usage may be in msg.get("usage", {})
          pass

  Option B — Count from the raw client (simpler for this lab):
      Monkey-patch or wrap get_llm_client() to intercept responses and
      accumulate usage. Or simply count tokens manually using
      tiktoken if AutoGen doesn't expose usage cleanly.

  Minimum requirement: record a non-zero total_tokens. Explain your
  counting method in the `notes` field of PlanningResult.

UserProxyAgent settings to use
--------------------------------
  human_input_mode="NEVER"     — fully autonomous, no keyboard prompts
  max_consecutive_auto_reply=0 — terminate after the first assistant reply
                                 (you want exactly one reply per conversation)
  is_termination_msg=lambda m: True   — ensures the conversation stops
                                        after the first message

System message guidance
-----------------------
Keep the same prompts as the LangGraph implementation — this ensures any
differences in output quality or token count are attributable to the
framework, not to different prompts.

JSON parsing
------------
AutoGen agents sometimes wrap JSON in markdown fences even when told not
to. Apply the same fence-stripping logic as the LangGraph implementation:

    text = text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()

notes field
-----------
Record at least two concrete observations about the AutoGen developer
experience compared to LangGraph. For example:
  - How many lines of framework setup code were required?
  - Did the agent respect the system_message reliably?
  - Was token usage easy or difficult to extract?
  - Did the conversation pattern feel natural for a sequential task?

Signature
---------
def run(requirement: str) -> PlanningResult
"""

import json
from dotenv import load_dotenv
load_dotenv()
import sys
import pysqlite3

sys.modules["sqlite3"] = pysqlite3

import autogen
from llm_client import get_autogen_llm_config
from shared_types import PlanningResult, TaskSpec


_SPEC_SYSTEM = """You are a senior software architect and Product Manager.
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

_DECOMPOSE_SYSTEM = """You are a technical project manager. Given a technical specification,
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


def _strip_fences(text: str) -> str:
        return text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()


def _extract_usage(messages: list[dict]) -> tuple[int, int, int]:
        prompt = completion = total = 0
        for msg in messages:
                usage = msg.get("usage") if isinstance(msg, dict) else None
                if usage:
                        prompt += int(usage.get("prompt_tokens", 0) or 0)
                        completion += int(usage.get("completion_tokens", 0) or 0)
                        total += int(usage.get("total_tokens", 0) or 0)
        if total == 0:
                total = prompt + completion
        return prompt, completion, total


def run(requirement: str) -> PlanningResult:
    """
    Run the PM planning step using AutoGen.

    Parameters
    ----------
    requirement : str
        The raw user requirement.

    Returns
    -------
    PlanningResult
        Spec, tasks, token usage, and implementation notes.
    """
    llm_config = get_autogen_llm_config()

    spec_writer = autogen.AssistantAgent(
        name="spec_writer",
        system_message=_SPEC_SYSTEM,
        llm_config=llm_config,
    )
    spec_user = autogen.UserProxyAgent(
        name="spec_user",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        is_termination_msg=lambda _: True,
    )
    spec_user.initiate_chat(spec_writer, message=f"Requirement: {requirement}")
    spec_text = spec_writer.last_message()["content"]

    task_writer = autogen.AssistantAgent(
        name="task_writer",
        system_message=_DECOMPOSE_SYSTEM,
        llm_config=llm_config,
    )
    task_user = autogen.UserProxyAgent(
        name="task_user",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        is_termination_msg=lambda _: True,
    )
    task_user.initiate_chat(task_writer, message=f"Technical specification:\n\n{spec_text}")
    tasks_text = task_writer.last_message()["content"]

    prompt_tokens, completion_tokens, total_tokens = _extract_usage(
        spec_writer.chat_messages[spec_user] + task_writer.chat_messages[task_user]
    )

    tasks = json.loads(_strip_fences(tasks_text))["tasks"]
    return {
        "framework": "autogen",
        "tech_spec": spec_text,
        "tasks": tasks,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "llm_calls": 2,
        "notes": (
            "AutoGen required explicit agent and proxy setup for two single-turn conversations. "
            "Token usage was extracted from chat history usage metadata, which was less direct than raw API responses."
        ),
    }


if __name__ == "__main__":
    from shared_types import BENCHMARK_REQUIREMENT
    result = run(BENCHMARK_REQUIREMENT)
    print(f"[AutoGen] {result['llm_calls']} calls | "
          f"{result['total_tokens']} tokens | "
          f"{len(result['tasks'])} tasks")
    if result["notes"]:
        print(f"Notes: {result['notes']}")
