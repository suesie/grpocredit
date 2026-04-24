"""Cascade orchestrator — wires Stage 0 / 1 / 2 (+ optional CUSUM) together.

Two modes of use:

**Offline (sprint Days 2–3).**
    `score_all_boundaries(...)` returns *every* candidate boundary with its
    s1, s2, cusum_abs, and stage_stopped_at annotations filled in — no
    budget-aware selection, no probe rollouts. Downstream oracle scripts
    compute ρ(s_2, Var(Q^π)) etc. across the full candidate set.

**Online (main training).**
    `select_probes(...)` takes a budget and returns the boundaries to probe
    plus per-boundary probe counts. Probes at top-30% of Stage-1 survivors
    by s_2, with ε-random hedge (§3.2.3 coverage clause).

Selection-bias safety: lookahead continuations from Stage 2 are stored on
the result for diagnostics but are *not* recycled as probe rollouts. Callers
must run fresh probes — which the online trainer does via
`backend.forced_action_rollouts` or ordinary `continue_from_prefixes`.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any

from grpocredit.common.config import CascadeConfig
from grpocredit.common.types import Boundary, RolloutResult, Trajectory
from grpocredit.rollout.vllm_runner import RolloutBackend
from grpocredit.voi.cusum_aux import CusumScorer
from grpocredit.voi.stage0_group_filter import Stage0Result, stage0_group_filter
from grpocredit.voi.stage1_entropy import Stage1Scorer
from grpocredit.voi.stage2_semantic import Stage2Scorer

log = logging.getLogger(__name__)


@dataclass
class PivotDecision:
    boundary: Boundary
    probe_count: int
    reason: str  # 'voi_top' | 'epsilon_random' | 'stage2_fallback' | 'offline'


@dataclass
class CascadeResult:
    trajectory_id: str
    stage0: Stage0Result | None
    all_boundaries: list[Boundary]
    stage1_survivors: list[Boundary] = field(default_factory=list)
    stage2_survivors: list[Boundary] = field(default_factory=list)
    decisions: list[PivotDecision] = field(default_factory=list)
    lookahead_rollouts: dict[int, list[RolloutResult]] = field(default_factory=dict)


@dataclass
class CascadeOrchestrator:
    config: CascadeConfig
    stage1: Stage1Scorer
    stage2: Stage2Scorer
    cusum: CusumScorer | None = None

    @classmethod
    def from_config(cls, config: CascadeConfig) -> CascadeOrchestrator:
        return cls(
            config=config,
            stage1=Stage1Scorer(config.stage1),
            stage2=Stage2Scorer(config.stage2),
            cusum=CusumScorer(config.cusum) if config.cusum.enabled else None,
        )

    # ── offline path ─────────────────────────────────────────────────
    def score_all_boundaries(
        self,
        backend: RolloutBackend,
        trajectory: Trajectory,
        boundaries: list[Boundary],
        *,
        run_stage2: bool = True,
        reference_logprobs: list[float] | None = None,
        seed_offset: int = 0,
    ) -> CascadeResult:
        """Score every candidate boundary with s1 / s2 / CUSUM; no filtering."""
        if not boundaries:
            return CascadeResult(
                trajectory_id=trajectory.trajectory_id,
                stage0=None,
                all_boundaries=[],
            )

        self.stage1.score(trajectory, boundaries)

        lookahead_rollouts: dict[int, list[RolloutResult]] = {}
        if run_stage2 and self.config.stage2.enabled:
            lookahead_rollouts = self.stage2.run_lookaheads(
                backend, trajectory, boundaries, seed_offset=seed_offset
            )
            self.stage2.score(boundaries, lookahead_rollouts)

        if self.cusum is not None and reference_logprobs is not None:
            self.cusum.score_boundaries(trajectory, boundaries, reference_logprobs)

        return CascadeResult(
            trajectory_id=trajectory.trajectory_id,
            stage0=None,
            all_boundaries=boundaries,
            lookahead_rollouts=lookahead_rollouts,
        )

    # ── online path (used by main trainer in Week 1+) ────────────────
    def select_probes(
        self,
        backend: RolloutBackend,
        group: list[Trajectory],
        boundaries_per_trajectory: dict[str, list[Boundary]],
        *,
        probe_budget: int,
        rollouts_per_probe: int = 8,
        seed_offset: int = 0,
    ) -> list[CascadeResult]:
        """Budget-aware probe selection for a full group of G trajectories."""
        s0 = stage0_group_filter(group)
        results: list[CascadeResult] = []

        if not s0.kept:
            # Stage 0 kills the group — no probes, no advantages needed beyond trajectory-level.
            for t in group:
                results.append(
                    CascadeResult(
                        trajectory_id=t.trajectory_id,
                        stage0=s0,
                        all_boundaries=boundaries_per_trajectory.get(t.trajectory_id, []),
                    )
                )
            return results

        # Stage 1 + Stage 2 on each trajectory's boundaries.
        all_decisions: list[tuple[Trajectory, Boundary]] = []
        per_traj: dict[str, CascadeResult] = {}
        for t in group:
            bds = boundaries_per_trajectory.get(t.trajectory_id, [])
            self.stage1.score(t, bds)
            s1_survivors = self.stage1.filter_top(bds)
            if s1_survivors and self.config.stage2.enabled:
                la = self.stage2.run_lookaheads(backend, t, s1_survivors, seed_offset=seed_offset)
                self.stage2.score(s1_survivors, la)
                s2_survivors = self.stage2.filter_top(s1_survivors)
            else:
                la = {}
                s2_survivors = s1_survivors

            cr = CascadeResult(
                trajectory_id=t.trajectory_id,
                stage0=s0,
                all_boundaries=bds,
                stage1_survivors=s1_survivors,
                stage2_survivors=s2_survivors,
                lookahead_rollouts=la,
            )
            per_traj[t.trajectory_id] = cr
            results.append(cr)
            for b in s2_survivors:
                all_decisions.append((t, b))

        # Budget allocation. Reserve ε of the budget for uniform-random boundaries
        # across the group (coverage hedge), split the rest evenly among the top s_2.
        eps = self.config.stage2.epsilon_random
        hedge_budget = int(round(eps * probe_budget))
        core_budget = probe_budget - hedge_budget

        # Core: top-k by s_2, equal share of core_budget.
        all_decisions.sort(key=lambda tb: tb[1].s2 or 0.0, reverse=True)
        n_core = min(len(all_decisions), max(1, core_budget // max(1, rollouts_per_probe)))
        for i, (t, b) in enumerate(all_decisions):
            cr = per_traj[t.trajectory_id]
            if i < n_core:
                b.selected = True
                cr.decisions.append(
                    PivotDecision(boundary=b, probe_count=rollouts_per_probe, reason="voi_top")
                )

        # Hedge: uniformly sample boundaries not already selected across the group.
        if hedge_budget > 0:
            rng = random.Random(seed_offset + 991)
            pool = [
                (t, b)
                for t, bs in [(tt, boundaries_per_trajectory.get(tt.trajectory_id, [])) for tt in group]
                for b in bs
                if not b.selected
            ]
            rng.shuffle(pool)
            n_hedge = min(len(pool), max(1, hedge_budget // max(1, rollouts_per_probe)))
            for (t, b) in pool[:n_hedge]:
                cr = per_traj[t.trajectory_id]
                b.selected = True
                cr.decisions.append(
                    PivotDecision(
                        boundary=b, probe_count=rollouts_per_probe, reason="epsilon_random"
                    )
                )

        return results
