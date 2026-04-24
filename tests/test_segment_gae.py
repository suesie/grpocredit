from __future__ import annotations

import numpy as np

from grpocredit.advantage.segment_gae import compute_segment_advantages


def test_no_interior_pivots_reduces_to_grpo() -> None:
    res = compute_segment_advantages(
        reward=1.0,
        baseline=0.3,
        trajectory_length=10,
        pivots=[],
    )
    # With no interior pivots, the single segment from 0 → T gets (r - r_bar)
    # distributed uniformly across all T tokens.
    per_token_expected = (1.0 - 0.3) / 10
    assert np.allclose(res.advantages, per_token_expected)
    assert res.mass_residual < 1e-6


def test_mass_conservation_with_interior_pivot() -> None:
    pivots = [(5, 0.6, 8)]  # one pivot at position 5, v_hat=0.6, M=8 rollouts
    res = compute_segment_advantages(
        reward=1.0,
        baseline=0.2,
        trajectory_length=10,
        pivots=pivots,
        shrinkage_mode="none",
    )
    # Deltas: V(5) - V(0) = 0.6 - 0.2 = 0.4, V(10) - V(5) = 1.0 - 0.6 = 0.4
    # Σ = 0.8 = (r - r_bar). mass_residual should be ~0 under 'none' shrinkage.
    assert res.mass_residual < 1e-6
    # First-half tokens should receive 0.4 / 5 = 0.08 per token
    assert np.allclose(res.advantages[:5], 0.08)
    assert np.allclose(res.advantages[5:], 0.08)


def test_shrinkage_reduces_interior_deltas_at_low_M() -> None:
    pivots = [(5, 0.6, 1)]  # single rollout at the pivot → strong shrinkage
    res_no = compute_segment_advantages(
        reward=1.0, baseline=0.2, trajectory_length=10, pivots=pivots, shrinkage_mode="none"
    )
    res_js = compute_segment_advantages(
        reward=1.0, baseline=0.2, trajectory_length=10, pivots=pivots,
        shrinkage_mode="james_stein", tau=4.0,
    )
    # Shrunk interior deltas must be strictly smaller in magnitude
    assert abs(res_js.segment_deltas[0]) < abs(res_no.segment_deltas[0])
    # Mass residual under shrinkage is nonzero (expected)
    assert res_js.mass_residual > 0.0
