"""Windowed-CUSUM auxiliary signal (§3.2.4).

Per-step implicit reward:
    r_φ(y_t | y_<t) = β · log [π_φ(y_t | y_<t) / π_ref(y_t | y_<t)]

Detection: δ_t = r_φ(y_t) − mean(r_φ) over a window of size W (default 15).

Selection: rank-based — pick the top-k% of |δ_t| across the group. No fixed
thresholds (they are scale-dependent and drift).

Status per proposal: this is *optional* — only stacked if an offline
rank-correlation diagnostic (§3.2.5 gate) shows ρ(|δ|, Var(Q^π)) > ρ(H_token)
alone. This module implements the signal; whether it's used is decided by
`configs/proposed/voi_full_cusum.yaml` after the sprint gate passes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from grpocredit.common.config import CusumConfig
from grpocredit.common.types import Boundary, Trajectory

log = logging.getLogger(__name__)


@dataclass
class CusumScorer:
    config: CusumConfig
    beta: float = 0.04  # matches default KL penalty in training config

    def implicit_rewards(
        self,
        trajectory: Trajectory,
        reference_logprobs: list[float],
    ) -> np.ndarray:
        """r_φ(y_t) = β · (logπ_φ(y_t) − logπ_ref(y_t))."""
        if len(reference_logprobs) != len(trajectory.logprobs):
            n = min(len(reference_logprobs), len(trajectory.logprobs))
            if n == 0:
                return np.zeros(0)
            log.warning(
                "CusumScorer: reference_logprobs length %d != trajectory.logprobs %d; "
                "truncating to %d",
                len(reference_logprobs),
                len(trajectory.logprobs),
                n,
            )
            trajectory_lp = np.asarray(trajectory.logprobs[:n])
            ref_lp = np.asarray(reference_logprobs[:n])
        else:
            trajectory_lp = np.asarray(trajectory.logprobs)
            ref_lp = np.asarray(reference_logprobs)
        return self.beta * (trajectory_lp - ref_lp)

    def windowed_delta(self, implicit_rewards: np.ndarray) -> np.ndarray:
        """δ_t = r_φ(y_t) − mean(r_φ) over a trailing window of size W."""
        r = np.asarray(implicit_rewards, dtype=float)
        n = len(r)
        if n == 0:
            return r
        W = max(1, self.config.window_size)
        # trailing mean: include current token
        cumsum = np.concatenate([[0.0], np.cumsum(r)])
        window_means = np.empty(n, dtype=float)
        for t in range(n):
            lo = max(0, t - W + 1)
            window_means[t] = (cumsum[t + 1] - cumsum[lo]) / (t + 1 - lo)
        return r - window_means

    def score_boundaries(
        self,
        trajectory: Trajectory,
        boundaries: list[Boundary],
        reference_logprobs: list[float],
    ) -> list[Boundary]:
        rewards = self.implicit_rewards(trajectory, reference_logprobs)
        deltas = self.windowed_delta(rewards)
        abs_deltas = np.abs(deltas)
        if len(abs_deltas) == 0:
            for b in boundaries:
                b.cusum_abs = 0.0
            return boundaries
        for b in boundaries:
            idx = max(0, min(b.token_position, len(abs_deltas) - 1))
            b.cusum_abs = float(abs_deltas[idx])
        return boundaries
