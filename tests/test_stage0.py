from __future__ import annotations

from grpocredit.common.types import Trajectory
from grpocredit.voi.stage0_group_filter import stage0_group_filter


def _mk(correct: bool) -> Trajectory:
    return Trajectory(
        trajectory_id="x",
        prompt_id="p",
        prompt="",
        prompt_token_ids=[],
        response_text="",
        token_ids=[1, 2, 3],
        logprobs=[-1.0, -1.0, -1.0],
        token_entropies=[0.5, 0.5, 0.5],
        reward=1.0 if correct else 0.0,
        correct=correct,
        ground_truth_answer="42",
    )


def test_all_correct_group_dropped() -> None:
    group = [_mk(True) for _ in range(4)]
    res = stage0_group_filter(group)
    assert res.kept is False
    assert res.success_rate == 1.0


def test_all_wrong_group_dropped() -> None:
    group = [_mk(False) for _ in range(4)]
    res = stage0_group_filter(group)
    assert res.kept is False
    assert res.success_rate == 0.0


def test_intermediate_group_kept() -> None:
    group = [_mk(i % 2 == 0) for i in range(4)]
    res = stage0_group_filter(group)
    assert res.kept is True
    assert 0.0 < res.success_rate < 1.0


def test_empty_group_dropped() -> None:
    res = stage0_group_filter([])
    assert res.kept is False
    assert res.group_size == 0
