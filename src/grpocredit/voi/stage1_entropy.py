"""Stage 1 — token-entropy × w_pos at syntactic boundaries.

Cost: zero extra compute — entropies are read off the per-step logprobs that
vLLM already returns during generation. Plan §3.2.2:
    s_1(b) = H(π(·|s_b)) · w_pos(t_b)

`H(π)` uses Shannon entropy by default (matches the VoI variance derivation —
linear in H, not √H). Optional `collision_complement` = 1 − ‖π‖² as an
alternative cheap proxy.

w_pos is one of:
    - 'tent':  w(t) = 1 − |2t/T − 1|   → peaks at T/2, zero at ends
    - 'gaussian': w(t) ∝ exp(−0.5·((t/T − 0.5)/σ)²)
    - 'uniform': w ≡ 1   → ablation that removes the position prior
    - 'lookup': read from a CSV of (decile, weight) — used after position_curve
                validates the empirical mid-peak on the oracle slice
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from grpocredit.common.config import Stage1Config
from grpocredit.common.types import Boundary, Trajectory


def token_entropy(logprobs: dict[int, float] | list[float]) -> float:
    """Shannon entropy over a (partial) logprob distribution.

    Accepts either:
      - dict {token_id: logprob}, treated as the top-k returned by vLLM
      - list of logprobs (already for specific tokens); assumed to cover the mass
    """
    if isinstance(logprobs, dict):
        lps = list(logprobs.values())
    else:
        lps = list(logprobs)
    if not lps:
        return 0.0
    h = 0.0
    for lp in lps:
        p = math.exp(lp)
        if p > 0:
            h -= p * lp
    return h


def collision_complement(logprobs: dict[int, float] | list[float]) -> float:
    """1 − ‖π‖² computed from partial top-k logprobs."""
    if isinstance(logprobs, dict):
        lps = list(logprobs.values())
    else:
        lps = list(logprobs)
    if not lps:
        return 0.0
    sq = sum(math.exp(2 * lp) for lp in lps)
    return 1.0 - sq


def _tent(frac: float) -> float:
    return max(0.0, 1.0 - abs(2.0 * frac - 1.0))


def _gauss(frac: float, sigma: float) -> float:
    return math.exp(-0.5 * ((frac - 0.5) / sigma) ** 2)


@dataclass
class Stage1Scorer:
    config: Stage1Config
    _lookup: list[tuple[float, float]] | None = None

    def __post_init__(self) -> None:
        if self.config.w_pos_shape == "lookup":
            if not self.config.w_pos_lookup_path:
                raise ValueError("w_pos_shape='lookup' requires w_pos_lookup_path")
            self._lookup = self._load_lookup(self.config.w_pos_lookup_path)

    @staticmethod
    def _load_lookup(path: str) -> list[tuple[float, float]]:
        """CSV with columns `decile,weight` (weights normalised to [0, 1] max)."""
        import csv

        rows: list[tuple[float, float]] = []
        with Path(path).open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append((float(row["decile"]), float(row["weight"])))
        rows.sort()
        max_w = max(w for _, w in rows) if rows else 1.0
        return [(d, w / max_w) for d, w in rows]

    def w_pos(self, token_position: int, trajectory_length: int) -> float:
        if trajectory_length <= 0:
            return 0.0
        frac = token_position / trajectory_length
        shape = self.config.w_pos_shape
        if shape == "uniform":
            return 1.0
        if shape == "tent":
            return _tent(frac)
        if shape == "gaussian":
            return _gauss(frac, self.config.w_pos_gaussian_sigma)
        if shape == "lookup":
            assert self._lookup is not None
            # piecewise linear interp over decile midpoints
            pts = self._lookup
            if frac <= pts[0][0]:
                return pts[0][1]
            if frac >= pts[-1][0]:
                return pts[-1][1]
            for (x0, y0), (x1, y1) in zip(pts, pts[1:], strict=False):
                if x0 <= frac <= x1:
                    t = (frac - x0) / max(x1 - x0, 1e-9)
                    return y0 + t * (y1 - y0)
            return 0.0
        raise ValueError(f"Unknown w_pos_shape: {shape}")

    def h_token_at(self, trajectory: Trajectory, token_position: int) -> float:
        """Read the per-step entropy computed during generation.

        `token_entropies[k]` is the entropy of π(·|prefix[:k]) — the distribution
        that sampled token_ids[k]. For a boundary at position `k`, π(·|s_b) is
        exactly this distribution (s_b = prefix[:k]).
        """
        ents = trajectory.token_entropies
        if not ents:
            return 0.0
        idx = max(0, min(token_position, len(ents) - 1))
        return float(ents[idx])

    def score(
        self, trajectory: Trajectory, boundaries: list[Boundary]
    ) -> list[Boundary]:
        T = trajectory.length
        for b in boundaries:
            h = self.h_token_at(trajectory, b.token_position)
            wp = self.w_pos(b.token_position, T)
            b.h_token = h
            b.w_pos = wp
            b.s1 = h * wp
        return boundaries

    def filter_top(
        self, boundaries: list[Boundary], keep_top_pct: float | None = None
    ) -> list[Boundary]:
        """Return the top-`keep_top_pct` fraction of boundaries by s1 (desc)."""
        if not boundaries:
            return []
        p = keep_top_pct if keep_top_pct is not None else self.config.keep_top_pct
        if p >= 1.0:
            return list(boundaries)
        k = max(1, int(round(len(boundaries) * p)))
        sorted_b = sorted(boundaries, key=lambda b: b.s1 or 0.0, reverse=True)
        survivors = sorted_b[:k]
        survivor_ids = {id(b) for b in survivors}
        for b in boundaries:
            if id(b) not in survivor_ids:
                b.stage_stopped_at = 1
        return survivors
