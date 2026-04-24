"""Stage 0 — success-rate group filter (DAPO dynamic sampling).

Drops groups where the success rate is 0/G (hopeless) or G/G (trivial); keeps
intermediate groups for Stage 1. This is the cheap VoI=0 screen (proposal
§3.2.1). Operates on a `Trajectory` group — a batch of G trajectories from
the same prompt — and returns the group only if intermediate.
"""

from __future__ import annotations

from dataclasses import dataclass

from grpocredit.common.types import Trajectory


@dataclass
class Stage0Result:
    kept: bool
    success_rate: float
    group_size: int


def stage0_group_filter(
    trajectories: list[Trajectory],
    *,
    trivial_threshold: float = 1.0,
    hopeless_threshold: float = 0.0,
) -> Stage0Result:
    """Return a decision; caller drops the group if `kept == False`.

    - `trivial_threshold = 1.0` → groups where every trajectory is correct drop.
    - `hopeless_threshold = 0.0` → groups where no trajectory is correct drop.
    - Intermediate 0 < p < 1 are kept.
    """
    g = len(trajectories)
    if g == 0:
        return Stage0Result(kept=False, success_rate=0.0, group_size=0)
    p = sum(1 for t in trajectories if t.correct) / g
    kept = hopeless_threshold < p < trivial_threshold
    return Stage0Result(kept=kept, success_rate=p, group_size=g)
