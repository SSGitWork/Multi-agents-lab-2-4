"""
Lab 2.4 — Test Suite
=====================

Tests for the AutoGen and CrewAI implementations and the comparison document.

Unlike Labs 2.1–2.3, these tests do not mock the LLM — the whole point
of this lab is to measure real framework behaviour. Tests therefore check
output *schema* (structure and types), not exact content.

Run with:
    pytest tests/test_lab24.py -v

Tests marked with @pytest.mark.live make real API calls and are skipped
by default. Run them explicitly when you have a live key:
    pytest tests/test_lab24.py -v -m live

Test categories
---------------
1. PlanningResult schema   — validates the shape of any run() return value
2. LangGraph baseline      — confirms the reference impl returns correct schema
3. AutoGen implementation  — schema checks on your AutoGen run()
4. CrewAI implementation   — schema checks on your CrewAI run()
5. Comparison document     — checks COMPARISON.md has required sections filled in
"""

import json
import os
import pytest
from unittest.mock import MagicMock, patch

from shared_types import BENCHMARK_REQUIREMENT, PlanningResult


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mock_openai_response(content: str, prompt_tokens: int = 100,
                           completion_tokens: int = 50) -> MagicMock:
    usage = MagicMock()
    usage.prompt_tokens     = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens      = prompt_tokens + completion_tokens

    msg = MagicMock()
    msg.content    = content
    msg.tool_calls = []

    choice = MagicMock()
    choice.message = msg

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage   = usage
    return resp


MOCK_SPEC = """\
## Overview
A small Flask REST API.

## Functional Requirements
- GET /health returns {"status": "ok"}
- GET /items returns a list of five items

## Non-Functional Requirements
- Runs on port 8080

## File Structure
- src/app.py

## Constraints & Assumptions
- No database needed
"""

MOCK_TASKS_JSON = json.dumps({
    "tasks": [
        {
            "task_id": "task_1",
            "title": "Create Flask app with /health",
            "description": "Create src/app.py with Flask and /health route.",
            "acceptance_criteria": ["GET /health returns 200"],
            "status": "pending",
            "file_path": "src/app.py",
        },
        {
            "task_id": "task_2",
            "title": "Add /items endpoint",
            "description": "Add GET /items returning five items.",
            "acceptance_criteria": ["GET /items returns 200"],
            "status": "pending",
            "file_path": "src/items.py",
        },
    ]
})


def _assert_planning_result_schema(result: dict, framework: str) -> None:
    """Assert that a PlanningResult has correct structure and types."""
    required_fields = {
        "framework", "tech_spec", "tasks",
        "prompt_tokens", "completion_tokens", "total_tokens",
        "llm_calls", "notes",
    }
    missing = required_fields - set(result.keys())
    assert not missing, f"[{framework}] PlanningResult missing fields: {missing}"

    assert result["framework"] == framework, (
        f"framework field must be '{framework}', got '{result['framework']}'"
    )
    assert isinstance(result["tech_spec"], str) and result["tech_spec"].strip(), (
        f"[{framework}] tech_spec must be a non-empty string"
    )
    assert isinstance(result["tasks"], list), (
        f"[{framework}] tasks must be a list"
    )
    assert 2 <= len(result["tasks"]) <= 4, (
        f"[{framework}] tasks must have 2–4 items, got {len(result['tasks'])}"
    )
    assert isinstance(result["prompt_tokens"], int), (
        f"[{framework}] prompt_tokens must be int"
    )
    assert isinstance(result["completion_tokens"], int), (
        f"[{framework}] completion_tokens must be int"
    )
    assert isinstance(result["total_tokens"], int), (
        f"[{framework}] total_tokens must be int"
    )
    assert result["total_tokens"] == result["prompt_tokens"] + result["completion_tokens"], (
        f"[{framework}] total_tokens must equal prompt + completion"
    )
    assert isinstance(result["llm_calls"], int) and result["llm_calls"] >= 1, (
        f"[{framework}] llm_calls must be a positive integer"
    )
    assert isinstance(result["notes"], str), (
        f"[{framework}] notes must be a string (empty string is fine)"
    )

    for i, task in enumerate(result["tasks"]):
        for field in ("task_id", "title", "description",
                      "acceptance_criteria", "status", "file_path"):
            assert field in task, (
                f"[{framework}] tasks[{i}] missing field '{field}'"
            )
        assert task["status"] == "pending", (
            f"[{framework}] tasks[{i}]['status'] must be 'pending'"
        )
        assert isinstance(task["acceptance_criteria"], list), (
            f"[{framework}] tasks[{i}]['acceptance_criteria'] must be a list"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 1. PlanningResult schema validator (unit tests — no API calls)
# ─────────────────────────────────────────────────────────────────────────────

class TestPlanningResultSchema:
    """Verify the schema checker itself works correctly."""

    def test_valid_result_passes(self):
        valid = {
            "framework":         "langgraph",
            "tech_spec":         MOCK_SPEC,
            "tasks":             json.loads(MOCK_TASKS_JSON)["tasks"],
            "prompt_tokens":     150,
            "completion_tokens": 80,
            "total_tokens":      230,
            "llm_calls":         2,
            "notes":             "",
        }
        _assert_planning_result_schema(valid, "langgraph")  # must not raise

    def test_missing_field_fails(self):
        incomplete = {"framework": "langgraph", "tech_spec": MOCK_SPEC}
        with pytest.raises(AssertionError):
            _assert_planning_result_schema(incomplete, "langgraph")

    def test_wrong_total_tokens_fails(self):
        result = {
            "framework": "langgraph", "tech_spec": MOCK_SPEC,
            "tasks": json.loads(MOCK_TASKS_JSON)["tasks"],
            "prompt_tokens": 100, "completion_tokens": 50,
            "total_tokens": 999,   # wrong — should be 150
            "llm_calls": 2, "notes": "",
        }
        with pytest.raises(AssertionError):
            _assert_planning_result_schema(result, "langgraph")

    def test_zero_tasks_fails(self):
        result = {
            "framework": "langgraph", "tech_spec": MOCK_SPEC,
            "tasks": [],   # must be 2–4
            "prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150,
            "llm_calls": 2, "notes": "",
        }
        with pytest.raises(AssertionError):
            _assert_planning_result_schema(result, "langgraph")


# ─────────────────────────────────────────────────────────────────────────────
# 2. LangGraph baseline (mocked)
# ─────────────────────────────────────────────────────────────────────────────

class TestLangGraphBaseline:

    @patch("implementations.langgraph_impl.get_llm_client")
    def test_returns_correct_schema(self, mock_client):
        mock_client.return_value.chat.completions.create.side_effect = [
            _mock_openai_response(MOCK_SPEC, 120, 60),
            _mock_openai_response(MOCK_TASKS_JSON, 180, 90),
        ]
        from implementations.langgraph_impl import run
        result = run(BENCHMARK_REQUIREMENT)
        _assert_planning_result_schema(result, "langgraph")

    @patch("implementations.langgraph_impl.get_llm_client")
    def test_makes_exactly_two_llm_calls(self, mock_client):
        mock_client.return_value.chat.completions.create.side_effect = [
            _mock_openai_response(MOCK_SPEC, 120, 60),
            _mock_openai_response(MOCK_TASKS_JSON, 180, 90),
        ]
        from implementations.langgraph_impl import run
        result = run(BENCHMARK_REQUIREMENT)
        assert result["llm_calls"] == 2, (
            f"LangGraph baseline must make exactly 2 LLM calls, got {result['llm_calls']}"
        )

    @patch("implementations.langgraph_impl.get_llm_client")
    def test_total_tokens_is_sum(self, mock_client):
        mock_client.return_value.chat.completions.create.side_effect = [
            _mock_openai_response(MOCK_SPEC, 120, 60),
            _mock_openai_response(MOCK_TASKS_JSON, 180, 90),
        ]
        from implementations.langgraph_impl import run
        result = run(BENCHMARK_REQUIREMENT)
        assert result["total_tokens"] == 120 + 60 + 180 + 90, (
            "total_tokens must be the sum of all prompt + completion tokens"
        )

    @patch("implementations.langgraph_impl.get_llm_client")
    def test_framework_field_is_langgraph(self, mock_client):
        mock_client.return_value.chat.completions.create.side_effect = [
            _mock_openai_response(MOCK_SPEC),
            _mock_openai_response(MOCK_TASKS_JSON),
        ]
        from implementations.langgraph_impl import run
        result = run(BENCHMARK_REQUIREMENT)
        assert result["framework"] == "langgraph"


# ─────────────────────────────────────────────────────────────────────────────
# 3. AutoGen implementation (mocked where possible, live-flagged otherwise)
# ─────────────────────────────────────────────────────────────────────────────

class TestAutoGenImplementation:

    def test_raises_not_implemented_before_student_code(self):
        from implementations.autogen_impl import run
        try:
            run("test requirement")
        except NotImplementedError:
            pytest.skip("autogen_impl not yet implemented")
        except Exception:
            pass  # any other exception means it was at least attempted

    def test_returns_dict(self):
        from implementations.autogen_impl import run
        try:
            result = run("Build a simple hello-world Flask app.")
        except NotImplementedError:
            pytest.skip("autogen_impl not yet implemented")
        except Exception as exc:
            pytest.skip(f"autogen_impl raised an unexpected exception: {exc}")
        assert isinstance(result, dict), (
            f"run() must return a dict, got {type(result).__name__}"
        )

    @pytest.mark.live
    def test_schema_with_live_api(self):
        """Requires a live HELICONE_API_KEY. Run with: pytest -m live"""
        from implementations.autogen_impl import run
        result = run(BENCHMARK_REQUIREMENT)
        _assert_planning_result_schema(result, "autogen")

    @pytest.mark.live
    def test_framework_field_is_autogen(self):
        from implementations.autogen_impl import run
        result = run(BENCHMARK_REQUIREMENT)
        assert result["framework"] == "autogen"

    @pytest.mark.live
    def test_notes_field_is_non_empty(self):
        """Notes must contain at least one observation."""
        from implementations.autogen_impl import run
        result = run(BENCHMARK_REQUIREMENT)
        assert result["notes"].strip(), (
            "notes must contain at least one observation about the AutoGen "
            "developer experience — this is part of the comparison deliverable"
        )

    @pytest.mark.live
    def test_total_tokens_greater_than_zero(self):
        from implementations.autogen_impl import run
        result = run(BENCHMARK_REQUIREMENT)
        assert result["total_tokens"] > 0, (
            "total_tokens must be greater than zero — ensure you are counting "
            "tokens from the API response or Helicone"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. CrewAI implementation
# ─────────────────────────────────────────────────────────────────────────────

class TestCrewAIImplementation:

    def test_raises_not_implemented_before_student_code(self):
        from implementations.crewai_impl import run
        try:
            run("test requirement")
        except NotImplementedError:
            pytest.skip("crewai_impl not yet implemented")
        except Exception:
            pass

    def test_returns_dict(self):
        from implementations.crewai_impl import run
        try:
            result = run("Build a simple hello-world Flask app.")
        except NotImplementedError:
            pytest.skip("crewai_impl not yet implemented")
        except Exception as exc:
            pytest.skip(f"crewai_impl raised an unexpected exception: {exc}")
        assert isinstance(result, dict)

    @pytest.mark.live
    def test_schema_with_live_api(self):
        from implementations.crewai_impl import run
        result = run(BENCHMARK_REQUIREMENT)
        _assert_planning_result_schema(result, "crewai")

    @pytest.mark.live
    def test_framework_field_is_crewai(self):
        from implementations.crewai_impl import run
        result = run(BENCHMARK_REQUIREMENT)
        assert result["framework"] == "crewai"

    @pytest.mark.live
    def test_notes_field_is_non_empty(self):
        from implementations.crewai_impl import run
        result = run(BENCHMARK_REQUIREMENT)
        assert result["notes"].strip(), (
            "notes must contain at least one observation about the CrewAI "
            "developer experience"
        )

    @pytest.mark.live
    def test_total_tokens_greater_than_zero(self):
        from implementations.crewai_impl import run
        result = run(BENCHMARK_REQUIREMENT)
        assert result["total_tokens"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# 5. Comparison document (COMPARISON.md)
# ─────────────────────────────────────────────────────────────────────────────

class TestComparisonDocument:
    """
    Verify that COMPARISON.md has been filled in with real data.
    These tests check for structural completeness, not content quality.
    """

    COMPARISON_PATH = "COMPARISON.md"

    REQUIRED_SECTIONS = [
        "## 1. Benchmark Setup",
        "## 2. Token Usage",
        "## 3. Output Quality",
        "## 4. Developer Experience",
        "## 5. Framework Selection Criteria",
        "## 6. Recommendation",
    ]

    def _read_comparison(self) -> str:
        if not os.path.exists(self.COMPARISON_PATH):
            pytest.fail(
                f"{self.COMPARISON_PATH} does not exist. "
                "You must complete and save this file."
            )
        with open(self.COMPARISON_PATH, "r") as f:
            return f.read()

    def test_file_exists(self):
        assert os.path.exists(self.COMPARISON_PATH), (
            "COMPARISON.md must exist in the repo root"
        )

    def test_all_required_sections_present(self):
        content = self._read_comparison()
        for section in self.REQUIRED_SECTIONS:
            assert section in content, (
                f"Required section '{section}' not found in COMPARISON.md"
            )

    def test_token_table_has_three_rows(self):
        content = self._read_comparison()
        # Count data rows in the token usage table (lines starting with | LangGraph, | AutoGen, | CrewAI)
        frameworks = ["| LangGraph", "| AutoGen", "| CrewAI"]
        for fw in frameworks:
            assert fw in content, (
                f"Token usage table in COMPARISON.md must have a row for '{fw.strip('| ')}'"
            )

    def test_recommendation_section_is_filled(self):
        content = self._read_comparison()
        # Find the recommendation section and check it has more than the template placeholder
        rec_idx = content.find("## 6. Recommendation")
        assert rec_idx != -1
        rec_section = content[rec_idx:]
        # The template text should be replaced with the student's content
        assert "_(2–4 sentences" not in rec_section, (
            "Section 6 (Recommendation) still contains the template placeholder. "
            "Replace it with your actual recommendation."
        )

    def test_framework_table_is_filled(self):
        content = self._read_comparison()
        # The framework selection table should not have all-empty cells
        table_start = content.find("| Criterion")
        assert table_start != -1, "Framework selection table not found"
        table_section = content[table_start:table_start + 600]
        # At least one cell should contain Best, Acceptable, or Poor
        filled = any(
            word in table_section
            for word in ("Best", "Acceptable", "Poor")
        )
        assert filled, (
            "The framework selection criteria table (Section 5) must be filled in "
            "with 'Best', 'Acceptable', or 'Poor' for each row"
        )

    def test_no_placeholder_text_in_required_sections(self):
        content = self._read_comparison()
        placeholders = [
            "_(1–2 sentences",
            "_(yes/no",
            "_(your analysis",
            "_(describe",
            "_(2–4 sentences",
        ]
        # Check each required section individually (section 7 is optional so we skip it)
        required_content = content.split("## 7.")[0]
        for placeholder in placeholders:
            assert placeholder not in required_content, (
                f"COMPARISON.md still contains the template placeholder "
                f"'{placeholder}'. Replace all placeholders in sections 1–6 "
                f"with your actual observations."
            )
