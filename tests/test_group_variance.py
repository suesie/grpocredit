"""Unit tests for the §5 group-variance gate compute.

Covers:
- saturated-policy regime (all-correct groups → informative fraction = 0)
- failure-floor regime (all-wrong groups → informative fraction = 0)
- mixed regime (the actual interesting case)
- degenerate group sizes (< 2)
- non-binary rewards (numerical std but no all-correct/all-wrong tagging)
- empty input
- Qwen-Instruct saturation scenario from sft_warmup_plan.md §1
- DeepSeekMath SFT'd target band scenario from sft_warmup_plan.md §3.A
"""

from __future__ import annotations

import math

import numpy as np

from grpocredit.oracle.group_variance import compute_group_variance_report


def test_empty_input_is_zero_not_nan():
    r = compute_group_variance_report([])
    assert r.n_groups == 0
    assert r.fraction_informative == 0.0
    assert math.isfinite(r.fraction_informative)


def test_all_correct_groups_zero_informative():
    # The Qwen-Instruct case: pass@1 ≈ 0.95 → most groups all-correct.
    grouped = [[1.0] * 8 for _ in range(64)]
    r = compute_group_variance_report(grouped)
    assert r.n_groups == 64
    assert r.n_informative == 0
    assert r.fraction_informative == 0.0
    assert r.n_groups_all_correct == 64
    assert r.n_groups_all_wrong == 0


def test_all_wrong_groups_zero_informative():
    grouped = [[0.0] * 8 for _ in range(32)]
    r = compute_group_variance_report(grouped)
    assert r.n_informative == 0
    assert r.n_groups_all_wrong == 32
    assert r.n_groups_all_correct == 0
    assert r.fraction_informative == 0.0


def test_mixed_groups_informative():
    grouped = [
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
    ]
    r = compute_group_variance_report(grouped)
    assert r.n_groups == 4
    assert r.n_informative == 2
    assert r.fraction_informative == 0.5
    assert r.n_groups_all_correct == 1
    assert r.n_groups_all_wrong == 1


def test_qwen_instruct_saturation_scenario():
    """From sft_warmup_plan.md §1: pass@1 ≈ 0.95 on GSM8K with G=8 → ≈ 66 %
    degenerate groups. Simulate by drawing each rollout Bernoulli(0.95) and
    checking the fraction of all-correct groups roughly matches the plan's
    estimate. (Statistical, so we test a coarse upper bound.)"""
    rng = np.random.default_rng(0)
    n_groups, G, p = 1024, 8, 0.95
    grouped = (rng.random(size=(n_groups, G)) < p).astype(float).tolist()
    r = compute_group_variance_report(grouped)
    # P(all correct) = 0.95^8 ≈ 0.66; P(all wrong) ≈ 4e-11 negligible.
    p_all_correct = p**G
    expected_informative_frac = 1.0 - p_all_correct  # ≈ 0.34
    # Allow ±5pt tolerance for the empirical sample.
    assert abs(r.fraction_informative - expected_informative_frac) < 0.05, (
        f"expected ~{expected_informative_frac:.2f}, got {r.fraction_informative:.2f}"
    )
    # Critically: this saturation case FAILS the §5 gate (≥ 0.5).
    assert r.fraction_informative < 0.5, "Qwen-Instruct simulation should FAIL the gate"


def test_deepseek_sft_target_band_passes_gate():
    """From sft_warmup_plan.md §3.A target band: GSM8K pass@1 in [0.65, 0.80].
    Simulate p=0.7; the gate should PASS comfortably."""
    rng = np.random.default_rng(1)
    n_groups, G, p = 1024, 8, 0.7
    grouped = (rng.random(size=(n_groups, G)) < p).astype(float).tolist()
    r = compute_group_variance_report(grouped)
    # P(all correct) = 0.7^8 ≈ 0.058; P(all wrong) = 0.3^8 ≈ 6.6e-5.
    expected_degenerate_frac = p**G + (1 - p) ** G
    expected_informative_frac = 1.0 - expected_degenerate_frac  # ≈ 0.94
    assert abs(r.fraction_informative - expected_informative_frac) < 0.03
    assert r.fraction_informative > 0.5, "DeepSeek-SFT target band should PASS the gate"


def test_degenerate_group_size_one_is_non_informative():
    grouped = [[1.0], [0.0], [1.0, 0.0]]
    r = compute_group_variance_report(grouped)
    # Singleton groups have std=0 → non-informative; only the 3rd group counts.
    assert r.n_informative == 1
    assert r.n_groups == 3
    assert math.isclose(r.fraction_informative, 1.0 / 3.0)


def test_non_binary_rewards_std_only():
    grouped = [
        [0.3, 0.7, 0.5],         # informative (std > 0)
        [0.5, 0.5, 0.5],         # non-informative, but neither all-correct nor all-wrong
    ]
    r = compute_group_variance_report(grouped)
    assert r.n_informative == 1
    assert r.n_groups_all_correct == 0
    assert r.n_groups_all_wrong == 0
    assert math.isclose(r.fraction_informative, 0.5)


def test_summary_stats_match_numpy():
    grouped = [
        [1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 1.0],
    ]
    r = compute_group_variance_report(grouped)
    expected_mean = float(np.mean([np.mean(g) for g in grouped]))
    expected_std = float(np.mean([np.std(g) for g in grouped]))
    assert math.isclose(r.mean_group_reward_mean, expected_mean, rel_tol=1e-9)
    assert math.isclose(r.mean_group_reward_std, expected_std, rel_tol=1e-9)
