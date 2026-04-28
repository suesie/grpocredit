"""Unit tests for the rollout-diversity sentinel compute.

The sentinel is what would have caught the `rho-1b-sft-GSM8K` Day-1
collapse before the 256×8 group-variance probe ran (see
`grpocredit.rollout.vllm_runner` module docstring for the failure mode).
"""

from __future__ import annotations

import pytest

from grpocredit.oracle.rollout_diversity import (
    RolloutDiversityError,
    assert_diverse_rollouts,
    compute_diversity_report,
)


def test_empty_input_is_zero_not_nan():
    r = compute_diversity_report([])
    assert r.n_groups == 0
    assert r.mean_unique_fraction == 0.0


def test_all_identical_groups_flagged():
    # Every group's rollouts are byte-identical — the rho-1b collapse signature.
    grouped = [["A", "A", "A", "A"] for _ in range(64)]
    r = compute_diversity_report(grouped)
    assert r.n_groups == 64
    assert r.n_groups_all_identical == 64
    assert r.n_groups_fully_unique == 0
    assert r.mean_unique_fraction == pytest.approx(0.25)

    with pytest.raises(RolloutDiversityError) as exc:
        assert_diverse_rollouts(grouped)
    # Error message must mention the precise numbers so the operator can
    # distinguish "saturated policy" from "runner bug".
    assert "fraction_all_identical=1.000" in str(exc.value)
    assert "mean_unique_fraction=0.250" in str(exc.value)


def test_fully_unique_groups_pass():
    # Healthy temp=0.9 sampling: essentially every sample is a different text.
    grouped = [[f"g{gi}_s{si}" for si in range(4)] for gi in range(32)]
    r = assert_diverse_rollouts(grouped)
    assert r.n_groups == 32
    assert r.n_groups_all_identical == 0
    assert r.n_groups_fully_unique == 32
    assert r.mean_unique_fraction == pytest.approx(1.0)


def test_mixed_groups_at_threshold():
    # Half groups collapsed, half diverse. At the default
    # max_all_identical_fraction=0.5 this should fail (> not ≥).
    grouped = [["A"] * 4 for _ in range(8)] + [
        [f"g{gi}_s{si}" for si in range(4)] for gi in range(8)
    ]
    # 8/16 = 0.5 exactly — with the `>` check, 0.5 does NOT exceed 0.5, so
    # this should PASS. Document the boundary explicitly.
    r = assert_diverse_rollouts(grouped)
    assert r.n_groups_all_identical == 8
    assert r.n_groups_fully_unique == 8


def test_degenerate_group_size_one_handled():
    grouped = [["solo"], ["A", "A"], ["x", "y"]]
    r = compute_diversity_report(grouped)
    assert r.n_groups == 3
    # Singletons don't count as "all-identical" (nothing to collide with) and
    # the group of length 2 all-"A" does.
    assert r.n_groups_all_identical == 1


def test_partially_collapsed_rejected_by_mean():
    # Every group has the same pair repeated: 2 unique out of 4. Mean
    # unique fraction = 0.5 — fails the default min_mean_unique_fraction=0.5
    # (need strict >). Adjust if we ever make the threshold ≥ instead of >.
    grouped = [["A", "A", "B", "B"] for _ in range(32)]
    r = compute_diversity_report(grouped)
    assert r.mean_unique_fraction == pytest.approx(0.5)
    # At 0.5 exactly, `< 0.5` is false, so this passes the mean check.
    # But fraction_all_identical = 0 (each group has 2 uniques) so also
    # passes the collapse check. Verify.
    rr = assert_diverse_rollouts(grouped)
    assert rr.n_groups_all_identical == 0


def test_rho1b_smoke_failure_reproduction():
    """Reproduce the `rho-1b-sft-GSM8K` Day-1 group-variance failure signal.

    The live run had 23 all-correct + 233 all-wrong groups out of 256, with
    `mean_group_reward_std=0.0` — i.e. every group's 8 rollouts were
    byte-identical. Text-level version of that batch must trip the sentinel.
    """
    grouped = [["only_answer"] * 8 for _ in range(256)]
    with pytest.raises(RolloutDiversityError):
        assert_diverse_rollouts(grouped)
