"""TD-style segment advantages over a sparse set of VoI-selected pivots (§3.3.3).

Given V̂(s) estimates at pivots 0 = τ_0 < τ_1 < ... < τ_K ≤ T:
    d_k         = V̂(τ_k) − V̂(τ_{k−1})        — segment delta
    d_end       = r − V̂(τ_K)                  — final residual
    A_t (∀ t ∈ (τ_{k−1}, τ_k])  += d_k / (τ_k − τ_{k−1})     — uniform in-segment

Mass conservation (sample-level): Σ d_k + d_end = r − V̂(τ_0). With V̂(τ_0)
defaulting to the group mean r̄, this recovers the GRPO baseline at the root.
Per §3.3.3 this is a *biased reward-redistribution scheme* — not an unbiased
estimator of the true token-level advantage — so ablation vs VinePPO's
per-chunk advantage is the primary empirical check.

Pivot convention: a pivot at index k means the "state" V̂ is evaluated at
`token_ids[:k]` (prefix of length k). Each token `t` in `(τ_{k−1}, τ_k]`
receives the k-th segment delta (where t=τ_k is included in segment k).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from grpocredit.advantage.shrinkage import apply_shrinkage


@dataclass
class SegmentAdvantageResult:
    """Per-token advantage for one trajectory + diagnostics."""

    advantages: np.ndarray  # shape (T,)
    segment_deltas: list[float]
    end_delta: float
    pivot_positions: list[int]
    pivot_v_hats: list[float]
    pivot_sample_counts: list[int]
    mass_residual: float  # |Σd − (r − r̄)| — mass conservation check
    extra: dict = field(default_factory=dict)


def compute_segment_advantages(
    *,
    reward: float,
    baseline: float,
    trajectory_length: int,
    pivots: list[tuple[int, float, int]],
    shrinkage_mode: str = "james_stein",
    tau: float = 4.0,
    var_v_estimates: list[float] | None = None,
) -> SegmentAdvantageResult:
    """Compute per-token advantages.

    Parameters
    ----------
    reward : float
        Trajectory-level terminal reward `r`.
    baseline : float
        Baseline at τ_0 — typically the group-mean `r̄` (GRPO baseline).
    trajectory_length : int
        Number of generated tokens `T`.
    pivots : list of (position, v_hat, sample_count)
        Pivots at interior positions 1 ≤ τ_k ≤ T-1. Must be sorted ascending by position.
        The τ_0 = 0 pivot (with V̂ = baseline) and τ_K = T (with V̂ = reward) are
        added automatically.
    shrinkage_mode, tau : passed through to `apply_shrinkage`
    var_v_estimates : per-pivot Var(V̂) estimates, same length as `pivots`.
        Used only for `shrinkage_mode = 'se'`.
    """
    T = trajectory_length
    if T <= 0:
        return SegmentAdvantageResult(
            advantages=np.zeros(0),
            segment_deltas=[],
            end_delta=0.0,
            pivot_positions=[],
            pivot_v_hats=[],
            pivot_sample_counts=[],
            mass_residual=0.0,
        )

    # Build the full pivot list: (τ_0=0, baseline, ∞) ... (interior pivots) ... (τ_K=T, r, ∞).
    # Sample count ∞ at boundary pivots means "no shrinkage" there.
    full_positions = [0]
    full_vhats = [baseline]
    full_counts = [10**9]

    interior_positions: list[int] = []
    interior_vhats: list[float] = []
    interior_counts: list[int] = []
    for pos, v_hat, m in pivots:
        if pos <= 0 or pos >= T:
            continue  # skip boundary-duplicating pivots
        interior_positions.append(pos)
        interior_vhats.append(v_hat)
        interior_counts.append(m)

    # Ensure strictly increasing positions — dedup by taking the last value per position
    if interior_positions:
        seen: dict[int, tuple[float, int]] = {}
        for pos, vh, m in zip(interior_positions, interior_vhats, interior_counts, strict=False):
            seen[pos] = (vh, m)
        items = sorted(seen.items())
        interior_positions = [p for p, _ in items]
        interior_vhats = [vh for _, (vh, _) in items]
        interior_counts = [m for _, (_, m) in items]

    full_positions.extend(interior_positions)
    full_vhats.extend(interior_vhats)
    full_counts.extend(interior_counts)
    full_positions.append(T)
    full_vhats.append(reward)
    full_counts.append(10**9)

    # Compute segment deltas.
    seg_deltas_raw: list[float] = []
    for k in range(1, len(full_positions)):
        d = full_vhats[k] - full_vhats[k - 1]
        seg_deltas_raw.append(d)

    # Apply shrinkage to *interior* deltas only — the τ_0→τ_1 and τ_{K-1}→τ_K
    # deltas both touch ∞-sample boundary pivots, so shrinkage reduces to α·d
    # where α is driven by the interior pivot's M. Use min(M_left, M_right) as
    # the effective sample count for the delta.
    seg_deltas_shrunk: list[float] = []
    for k in range(1, len(full_positions)):
        m_left = full_counts[k - 1]
        m_right = full_counts[k]
        m_eff = min(m_left, m_right)
        var_v_k = 0.0
        if var_v_estimates is not None and 0 < k - 1 < len(var_v_estimates) + 1:
            idx = k - 1
            if 0 <= idx < len(var_v_estimates):
                var_v_k = float(var_v_estimates[idx])
        d = seg_deltas_raw[k - 1]
        # Only shrink interior deltas where at least one side is a finite-M pivot
        if m_eff >= 10**8:
            seg_deltas_shrunk.append(d)
        else:
            seg_deltas_shrunk.append(
                apply_shrinkage(d, m=m_eff, mode=shrinkage_mode, tau=tau, var_v=var_v_k)
            )

    # Assign to tokens uniformly within each segment.
    advantages = np.zeros(T, dtype=np.float32)
    for k in range(1, len(full_positions)):
        lo = full_positions[k - 1]
        hi = full_positions[k]
        seg_len = max(1, hi - lo)
        per_token = seg_deltas_shrunk[k - 1] / seg_len
        advantages[lo:hi] += per_token

    end_delta = seg_deltas_shrunk[-1]  # τ_{K-1} → τ_K = T segment, which is r − V̂(τ_{K-1})
    mass_residual = float(abs(sum(seg_deltas_shrunk) - (reward - baseline)))

    return SegmentAdvantageResult(
        advantages=advantages,
        segment_deltas=seg_deltas_shrunk,
        end_delta=end_delta,
        pivot_positions=full_positions,
        pivot_v_hats=full_vhats,
        pivot_sample_counts=full_counts,
        mass_residual=mass_residual,
    )
