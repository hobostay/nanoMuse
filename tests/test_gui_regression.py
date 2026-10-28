from __future__ import annotations

from pathlib import Path

import pytest

from nanomuse.config import Settings, TaskBudgetSettings
from nanomuse.evaluation.gui import load_cases, reference_llm, run_case, summarize
from nanomuse.llm import MockLLM
from nanomuse.llm import budget as budgets
from nanomuse.schema import LLMResponse

CASES = load_cases(Path(__file__).parent / "fixtures/gui_regression/cases.v1.json")


@pytest.mark.parametrize("case", CASES, ids=[case.id for case in CASES])
async def test_gui_reference_cases(case, settings: Settings, tmp_path: Path):
    result = await run_case(case, settings, tmp_path / case.id)
    assert result["passed"], result
    assert result["trace"] and result["budget"]["calls"]
    if case.id == "approval-deny":
        assert result["actions"] == 0


async def test_claiming_success_without_acting_fails_the_oracle(settings: Settings, tmp_path: Path):
    done = LLMResponse(
        content='Action: Done\n<tool_call>{"name":"mobile_use","arguments":{"action":"terminate","status":"success","text":"Done"}}</tool_call>'
    )
    result = await run_case(CASES[0], settings, tmp_path / "false-success", MockLLM([done]))
    assert result["status"] == "done" and not result["passed"]
    assert not result["checks"]["scene"] and not result["checks"]["values"]


async def test_wrong_text_fails_even_after_save(settings: Settings, tmp_path: Path):
    model = reference_llm(CASES[0])
    model.script[1] = LLMResponse(
        content='Action: Type\n<tool_call>{"name":"mobile_use","arguments":{"action":"type","text":"wrong"}}</tool_call>'
    )
    result = await run_case(CASES[0], settings, tmp_path / "wrong-value", model)
    assert result["checks"]["scene"] and not result["checks"]["values"]
    assert not result["passed"]


async def test_operator_inherits_parent_token_limit(settings: Settings, tmp_path: Path):
    token = budgets.begin(TaskBudgetSettings(max_total_tokens=1))
    try:
        result = await run_case(CASES[0], settings, tmp_path / "budget")
        assert result["status"] == "failed" and result["actions"] == 0
        assert result["budget"]["stopped"] == "token"
    finally:
        budgets.end(token)


def test_summary_handles_empty_runs_and_nearest_rank_p95():
    assert summarize([], mode="reference")["pass_rate"] == 0
    rows = [{"passed": i % 2 == 0, "elapsed_seconds": i, "budget": {}} for i in range(20)]
    report = summarize(rows, mode="reference")
    assert report["pass_rate"] == 0.5 and report["p95_seconds"] == 18
