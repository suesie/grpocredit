"""Noise-adaptive shrinkage for MC-estimated segment deltas (§3.3.4).

Two families:
  A) James–Stein-style:   α(M) = M / (M + τ)
     Applied as d' = α(M) · d. Large M → α→1 (no shrinkage); small M → α→0
     (strong shrinkage). τ≈4 tunes the crossover.
  B) SE-based:            d' = d / sqrt(1 + Var(V̂)/M)
     Shrinkage proportional to the standard error of V̂ at the pivot.

Both are load-bearing under heterogeneous per-pivot sample counts produced by
VoI waterfill allocation — see §3.3.4 note on variance doubling from
subtracting two noisy MC estimates.
"""

from __future__ import annotations


def james_stein_alpha(m: int, tau: float = 4.0) -> float:
    """α(M) = M / (M + τ)."""
    if m <= 0:
        return 0.0
    return m / (m + tau)


def se_shrinkage(delta: float, var_v: float, m: int) -> float:
    """d' = d / sqrt(1 + Var(V̂)/M). `var_v` is an external estimate of Var(V̂)."""
    if m <= 0:
        return 0.0
    import math

    denom = math.sqrt(1.0 + max(var_v, 0.0) / m)
    return delta / denom if denom > 0 else 0.0


def apply_shrinkage(
    delta: float,
    m: int,
    mode: str = "james_stein",
    tau: float = 4.0,
    var_v: float = 0.0,
) -> float:
    if mode == "none":
        return delta
    if mode == "james_stein":
        return delta * james_stein_alpha(m, tau=tau)
    if mode == "se":
        return se_shrinkage(delta, var_v=var_v, m=m)
    raise ValueError(f"Unknown shrinkage mode: {mode}")
