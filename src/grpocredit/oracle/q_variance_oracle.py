"""Offline Q^π-variance oracle (§3.2.5).

At each sampled boundary:
  1. Read the top-M first-actions by π(·|s_b) — with full-support coverage
     c = Σ_{a∈A_s} π(a|s_b).
  2. For each a ∈ A_s: force action a, run K terminal rollouts → Q̂^π(s, a).
  3. If coverage c < 0.9 and `include_tail_stratum`: run K' rollouts under
     a ∼ π(·|s_b, a ∉ A_s) → Q̂^π_tail(s). Combined via stratified sampling.
  4. V̂^π(s) = Σ_a π(a) Q̂^π(s, a) [+ (1-c) Q̂^π_tail]
     Var_{a∼π}(Q^π) = Σ_a π(a) (Q̂^π(s, a) − V̂^π(s))²
                      [+ (1-c) (Q̂^π_tail − V̂^π(s))² if tail included]

Do NOT use `max_a Q̂^π − mean_a Q̂^π` — it's a separation, not a variance (see
§3.2.5 caveat). The label on the reported variance is "head-truncated" if the
tail stratum is not run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from grpocredit.common.config import OracleConfig
from grpocredit.common.types import (
    Boundary,
    ForcedActionResult,
    OracleRecord,
    Trajectory,
)
from grpocredit.rollout.verifier import MathVerifier
from grpocredit.rollout.vllm_runner import RolloutBackend

log = logging.getLogger(__name__)


@dataclass
class OracleBoundaryRecord:
    """Compact per-boundary bundle used internally before we promote to OracleRecord."""

    boundary: Boundary
    trajectory: Trajectory
    forced_results: list[ForcedActionResult]
    tail_result: ForcedActionResult | None
    coverage_c: float


@dataclass
class QVarianceResult:
    records: list[OracleRecord]
    total_rollouts: int
    config: OracleConfig


def _compute_variance(
    pi_probs: list[float],
    q_hats: list[float],
    tail_p: float = 0.0,
    tail_q: float | None = None,
) -> tuple[float, float]:
    """Return (V̂, Var_{a∼π}(Q^π)). Head+optional-tail stratification."""
    assert len(pi_probs) == len(q_hats)
    arr_p = np.asarray(pi_probs, dtype=float)
    arr_q = np.asarray(q_hats, dtype=float)
    v_head = float(np.sum(arr_p * arr_q))
    if tail_q is not None and tail_p > 0.0:
        v_hat = v_head + tail_p * tail_q
    else:
        v_hat = v_head
    head_var = float(np.sum(arr_p * (arr_q - v_hat) ** 2))
    if tail_q is not None and tail_p > 0.0:
        var = head_var + tail_p * (tail_q - v_hat) ** 2
    else:
        var = head_var
    return v_hat, var


@dataclass
class QVarianceOracle:
    config: OracleConfig
    verifier: MathVerifier = field(default_factory=MathVerifier)

    def probe_one_boundary(
        self,
        backend: RolloutBackend,
        trajectory: Trajectory,
        boundary: Boundary,
        ground_truth: str,
        *,
        max_new_tokens: int,
        temperature: float = 0.9,
        seed_offset: int = 0,
    ) -> OracleBoundaryRecord:
        prefix_ids = trajectory.prompt_token_ids + trajectory.token_ids[: boundary.token_position]

        # 1. Read top-M first-actions π(·|s_b).
        probs = backend.policy_probs_at(prefix_ids, top_k=self.config.top_m_actions)
        if not probs:
            log.warning(
                "policy_probs_at returned empty at boundary %s:%d",
                trajectory.trajectory_id,
                boundary.boundary_idx,
            )
            return OracleBoundaryRecord(
                boundary=boundary, trajectory=trajectory,
                forced_results=[], tail_result=None, coverage_c=0.0
            )

        first_token_ids = [tid for tid, _ in probs]
        pi_probs = [p for _, p in probs]
        coverage_c = float(sum(pi_probs))

        # 2. For each a ∈ A_s: force a, run K terminal rollouts.
        rollouts = backend.forced_action_rollouts(
            prefix_token_ids=prefix_ids,
            first_token_ids=first_token_ids,
            n_per_action=self.config.rollouts_per_forced_action,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
            seed=seed_offset,
        )

        forced_results: list[ForcedActionResult] = []
        for (tid, p), rs in zip(probs, rollouts, strict=False):
            rewards = []
            response_texts = []
            n_correct = 0
            for r in rs:
                v = self.verifier.score(r.response_text, ground_truth)
                rewards.append(1.0 if v.correct else 0.0)
                response_texts.append(r.response_text)
                if v.correct:
                    n_correct += 1
            forced_results.append(
                ForcedActionResult(
                    boundary_id=f"{trajectory.trajectory_id}:{boundary.boundary_idx}",
                    first_token_id=tid,
                    first_token_str=backend.detokenize([tid]),
                    pi_first_token=float(p),
                    rewards=rewards,
                    response_texts=response_texts,
                    num_correct=n_correct,
                )
            )

        # 3. Optional sampled-tail stratum.
        tail_result: ForcedActionResult | None = None
        if (
            self.config.include_tail_stratum
            and coverage_c < self.config.coverage_threshold_for_tail
        ):
            tail_result = self._run_tail_stratum(
                backend=backend,
                prefix_ids=prefix_ids,
                first_token_ids=first_token_ids,
                ground_truth=ground_truth,
                trajectory_id=trajectory.trajectory_id,
                boundary_idx=boundary.boundary_idx,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                seed_offset=seed_offset + 7919,
            )

        return OracleBoundaryRecord(
            boundary=boundary,
            trajectory=trajectory,
            forced_results=forced_results,
            tail_result=tail_result,
            coverage_c=coverage_c,
        )

    def _run_tail_stratum(
        self,
        *,
        backend: RolloutBackend,
        prefix_ids: list[int],
        first_token_ids: list[int],
        ground_truth: str,
        trajectory_id: str,
        boundary_idx: int,
        max_new_tokens: int,
        temperature: float,
        seed_offset: int,
    ) -> ForcedActionResult:
        """Sample tail-actions from π(·|s_b, a ∉ A_s) and roll out.

        Approximation: sample first tokens via regular continuation, reject any
        that fall in A_s, and re-sample. For numerical simplicity and to keep
        the oracle interface self-contained we do the rejection in the loop
        — not perfectly IS-corrected, but matches the §3.1.3 operational rule.
        """
        blocked = set(first_token_ids)
        desired = self.config.tail_stratum_size
        collected_rewards: list[float] = []
        collected_texts: list[str] = []
        n_correct = 0
        attempts = 0
        max_attempts = desired * 8

        while len(collected_rewards) < desired and attempts < max_attempts:
            batch_n = min(desired - len(collected_rewards), 16)
            attempts += batch_n
            out = backend.continue_from_prefixes(
                prefix_token_ids=[prefix_ids],
                n_continuations=batch_n,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.95,
                seed=seed_offset + attempts,
            )[0]
            for r in out:
                first_tid = r.token_ids[0] if r.token_ids else -1
                if first_tid in blocked:
                    continue
                v = self.verifier.score(r.response_text, ground_truth)
                collected_rewards.append(1.0 if v.correct else 0.0)
                collected_texts.append(r.response_text)
                if v.correct:
                    n_correct += 1
        return ForcedActionResult(
            boundary_id=f"{trajectory_id}:{boundary_idx}",
            first_token_id=-1,
            first_token_str="<tail>",
            pi_first_token=float("nan"),
            rewards=collected_rewards,
            response_texts=collected_texts,
            num_correct=n_correct,
        )

    def finalise_record(self, obr: OracleBoundaryRecord) -> OracleRecord:
        """Aggregate a OracleBoundaryRecord → OracleRecord with V̂, Var."""
        boundary = obr.boundary
        traj = obr.trajectory
        if not obr.forced_results:
            return OracleRecord(
                boundary_id=f"{traj.trajectory_id}:{boundary.boundary_idx}",
                trajectory_id=traj.trajectory_id,
                boundary_idx=boundary.boundary_idx,
                token_position=boundary.token_position,
                relative_position=boundary.token_position / max(1, traj.length),
                trajectory_length=traj.length,
                coverage_c=obr.coverage_c,
                forced_action_results=[],
                tail_result=obr.tail_result,
                h_token=boundary.h_token,
                w_pos=boundary.w_pos,
                s1=boundary.s1,
                h_sem=boundary.h_sem,
                s2=boundary.s2,
                v_hat=None,
                var_q_pi=None,
            )

        pi_probs = [fr.pi_first_token for fr in obr.forced_results]
        q_hats = [fr.q_hat for fr in obr.forced_results]
        tail_p = max(0.0, 1.0 - obr.coverage_c) if obr.tail_result is not None else 0.0
        tail_q = obr.tail_result.q_hat if obr.tail_result is not None else None

        v_hat, var = _compute_variance(pi_probs, q_hats, tail_p=tail_p, tail_q=tail_q)

        return OracleRecord(
            boundary_id=f"{traj.trajectory_id}:{boundary.boundary_idx}",
            trajectory_id=traj.trajectory_id,
            boundary_idx=boundary.boundary_idx,
            token_position=boundary.token_position,
            relative_position=boundary.token_position / max(1, traj.length),
            trajectory_length=traj.length,
            coverage_c=obr.coverage_c,
            forced_action_results=obr.forced_results,
            tail_result=obr.tail_result,
            h_token=boundary.h_token,
            w_pos=boundary.w_pos,
            s1=boundary.s1,
            h_sem=boundary.h_sem,
            s2=boundary.s2,
            v_hat=v_hat,
            var_q_pi=var,
        )

    def run(
        self,
        backend: RolloutBackend,
        trajectory_boundaries: Iterable[tuple[Trajectory, Boundary, str]],
        *,
        max_new_tokens: int,
        temperature: float = 0.9,
        seed: int = 0,
    ) -> QVarianceResult:
        """Run the oracle over a list of (trajectory, boundary, ground_truth) triples."""
        records: list[OracleRecord] = []
        total = 0
        for i, (traj, b, gt) in enumerate(trajectory_boundaries):
            obr = self.probe_one_boundary(
                backend,
                traj,
                b,
                gt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                seed_offset=seed + i,
            )
            rec = self.finalise_record(obr)
            total += sum(len(fr.rewards) for fr in obr.forced_results)
            if obr.tail_result is not None:
                total += len(obr.tail_result.rewards)
            records.append(rec)
        return QVarianceResult(records=records, total_rollouts=total, config=self.config)
