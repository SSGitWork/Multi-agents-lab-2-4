"""
Lab 2.4 — Comparison Runner
=============================

Runs all three framework implementations against the benchmark requirement
and prints a side-by-side comparison table.

Usage
-----
    python run_comparison.py              # runs all three
    python run_comparison.py langgraph    # runs one framework only
    python run_comparison.py autogen
    python run_comparison.py crewai
"""

import sys
import json
from dotenv import load_dotenv
load_dotenv()

from shared_types import BENCHMARK_REQUIREMENT, PlanningResult


def run_one(framework: str, requirement: str) -> PlanningResult | None:
    try:
        if framework == "langgraph":
            from implementations.langgraph_impl import run
        elif framework == "autogen":
            from implementations.autogen_impl import run
        elif framework == "crewai":
            from implementations.crewai_impl import run
        else:
            print(f"Unknown framework: {framework}")
            return None
        print(f"Running {framework}...", flush=True)
        return run(requirement)
    except NotImplementedError:
        print(f"  [{framework}] Not yet implemented — skipping.")
        return None
    except Exception as exc:
        print(f"  [{framework}] ERROR: {exc}")
        return None


def print_comparison(results: list[PlanningResult]) -> None:
    if not results:
        print("No results to compare.")
        return

    print(f"\n{'='*70}")
    print("FRAMEWORK COMPARISON — PM Planning Step")
    print(f"Requirement: {BENCHMARK_REQUIREMENT[:60]}...")
    print(f"{'='*70}\n")

    headers = ["Framework", "LLM Calls", "Prompt Tok", "Completion Tok", "Total Tok", "Tasks"]
    col_w   = [12, 10, 11, 15, 10, 6]

    header_row = "  ".join(h.ljust(w) for h, w in zip(headers, col_w))
    print(header_row)
    print("-" * len(header_row))

    for r in results:
        row = [
            r["framework"],
            str(r["llm_calls"]),
            str(r["prompt_tokens"]),
            str(r["completion_tokens"]),
            str(r["total_tokens"]),
            str(len(r["tasks"])),
        ]
        print("  ".join(v.ljust(w) for v, w in zip(row, col_w)))

    print(f"\n{'─'*70}")
    print("TASK DECOMPOSITIONS\n")
    for r in results:
        print(f"[{r['framework'].upper()}]")
        for task in r["tasks"]:
            print(f"  {task['task_id']}  {task['title']}")
            print(f"        → {task['file_path']}")
        print()

    print(f"{'─'*70}")
    print("IMPLEMENTATION NOTES\n")
    for r in results:
        if r["notes"]:
            print(f"[{r['framework'].upper()}]")
            print(f"  {r['notes']}\n")


def save_results(results: list[PlanningResult]) -> None:
    path = "comparison_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {path}")
    print("Use these figures in your comparison document (COMPARISON.md).")


if __name__ == "__main__":
    target = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    frameworks = (
        ["langgraph", "autogen", "crewai"] if target == "all"
        else [target]
    )

    results = []
    for fw in frameworks:
        r = run_one(fw, BENCHMARK_REQUIREMENT)
        if r:
            results.append(r)

    if results:
        print_comparison(results)
        if len(results) > 1:
            save_results(results)
