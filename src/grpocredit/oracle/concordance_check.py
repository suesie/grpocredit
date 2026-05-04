"""Day 2A — embedding-variance vs reward-variance diagnostic.

At each boundary of each trajectory from *informative* prompts (prompts
where G rollouts showed mixed outcomes, i.e., 0 < pass_rate < 1):

  1. Run K=4 terminal continuations from the boundary prefix.
  2. Truncate each to ≤30 tokens → the "lookahead" L_i.
  3. Embed L_i via sentence-T5-base → ℝ^768.
  4. Compute embedding variance:
     - emb_var_cosine = 1 − mean(pairwise cosines)
  5. Score each full continuation with the verifier → binary reward.
  6. Compute reward_var = Var(rewards).

Primary metrics (selection-based, per-trajectory):
  - top1_agreement: fraction of trajectories where the boundary with max
    emb_var also has max reward_var.
  - overlap_at_2: Jaccard overlap of top-2 boundaries by emb_var vs top-2
    by reward_var, for trajectories with ≥4 boundaries.
  - kappa_emb: mean reward_var at top-1 emb_var boundary / mean reward_var
    across all boundaries in the same trajectory.  >1 = signal has
    selection value.

Secondary (legacy): Spearman ρ(emb_var, reward_var) across all boundaries.

Caveat: when the remaining tokens after a boundary are comparable to or
shorter than the lookahead length (30), the "preview" captures most of
the terminal response and the correlation is inflated.  The result
includes a `rho_cosine_long_only` subset restricted to boundaries where
remaining_tokens > 2 × lookahead_max_tokens to diagnose this.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats as sp_stats

from grpocredit.common.config import OracleConfig, Stage2Config
from grpocredit.common.types import Boundary, Trajectory
from grpocredit.common.utils import fisher_z_ci
from grpocredit.rollout.verifier import MathVerifier
from grpocredit.rollout.vllm_runner import RolloutBackend
from grpocredit.voi.stage2_semantic import Stage2Scorer

log = logging.getLogger(__name__)


# ── Dataclasses ─────────────────────────────────────────────────────


@dataclass
class EmbVarBoundaryRecord:
    """Per-boundary record of embedding variance vs reward variance."""

    trajectory_id: str
    boundary_idx: int
    token_position: int
    relative_position: float
    remaining_tokens: int
    n_samples: int
    emb_var_cosine: float  # 1 - mean(pairwise cosines)
    emb_var_trace: float  # trace(Cov(embeddings))
    reward_var: float  # Var of K binary terminal rewards (ddof=0)
    mean_reward: float
    n_correct: int
    lookahead_texts: list[str] = field(default_factory=list)
    terminal_texts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "boundary_idx": self.boundary_idx,
            "token_position": self.token_position,
            "relative_position": self.relative_position,
            "remaining_tokens": self.remaining_tokens,
            "n_samples": self.n_samples,
            "emb_var_cosine": self.emb_var_cosine,
            "emb_var_trace": self.emb_var_trace,
            "reward_var": self.reward_var,
            "mean_reward": self.mean_reward,
            "n_correct": self.n_correct,
            "lookahead_texts": self.lookahead_texts,
            "terminal_texts": self.terminal_texts,
        }


@dataclass
class EmbVarResult:
    """Aggregate result of the embedding-variance diagnostic."""

    records: list[EmbVarBoundaryRecord]
    # ── Selection metrics (per-trajectory, the primary decision signals) ──
    top1_agreement: float  # frac of trajs where argmax(emb_var) == argmax(reward_var)
    top1_n_trajectories: int  # trajectories with ≥2 boundaries used for top-1
    overlap_at_2: float  # Jaccard of top-2 by emb_var vs top-2 by reward_var
    overlap_at_2_n_trajectories: int  # trajectories with ≥4 boundaries
    kappa_emb: float  # mean(reward_var at best emb_var bd) / mean(reward_var across all bds)
    kappa_emb_n_trajectories: int
    # ── Spearman correlations (all boundaries, legacy) ──
    rho_cosine: float
    rho_trace: float
    rho_cosine_ci: tuple[float, float]
    rho_trace_ci: tuple[float, float]
    # Subset: boundaries with remaining > 2 * lookahead_max_tokens
    rho_cosine_long_only: float
    rho_trace_long_only: float
    n_long_boundaries: int
    # Metadata
    n_informative_prompts: int
    n_boundaries: int
    total_rollouts: int
    mean_remaining_tokens: float
    lookahead_max_tokens: int
    n_emb_samples: int  # rollouts used for embedding variance
    n_reward_samples: int  # rollouts used for reward variance


# ── Pure-compute helpers ────────────────────────────────────────────


def _pairwise_cosine_var(embeddings: np.ndarray) -> float:
    """1 - mean of upper-triangle pairwise cosine similarities.

    Assumes rows are already L2-normalised (Stage2Scorer.embed does this).
    """
    n = len(embeddings)
    if n < 2:
        return 0.0
    sim = embeddings @ embeddings.T
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return float(1.0 - np.mean(sim[mask]))


def _trace_cov(embeddings: np.ndarray) -> float:
    """Trace of the sample covariance matrix = sum of per-dim variances."""
    if len(embeddings) < 2:
        return 0.0
    return float(np.var(embeddings, axis=0, ddof=1).sum())


def _spearman(x: list[float], y: list[float]) -> float:
    arr_x = np.asarray(x, dtype=float)
    arr_y = np.asarray(y, dtype=float)
    mask = np.isfinite(arr_x) & np.isfinite(arr_y)
    if mask.sum() < 4:
        return float("nan")
    return float(sp_stats.spearmanr(arr_x[mask], arr_y[mask]).correlation)


def _truncate_text_first_tokens(
    text: str, tokenizer: Any, max_tokens: int
) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)


def _compute_selection_metrics(
    records: list[EmbVarBoundaryRecord],
) -> dict[str, float | int]:
    """Per-trajectory selection metrics for the cascade decision problem.

    Groups boundaries by trajectory_id, then computes:
      (a) top1_agreement — does argmax(emb_var) == argmax(reward_var)?
      (b) overlap@2 — Jaccard of top-2 sets
      (c) kappa_emb — reward_var at selected / mean reward_var
    """
    from collections import defaultdict

    by_traj: dict[str, list[EmbVarBoundaryRecord]] = defaultdict(list)
    for r in records:
        by_traj[r.trajectory_id].append(r)

    # (a) Top-1 agreement: trajectories with ≥2 boundaries
    top1_hits = 0
    top1_total = 0
    for recs in by_traj.values():
        if len(recs) < 2:
            continue
        top1_total += 1
        best_emb_idx = max(range(len(recs)), key=lambda i: recs[i].emb_var_cosine)
        best_rew_idx = max(range(len(recs)), key=lambda i: recs[i].reward_var)
        if best_emb_idx == best_rew_idx:
            top1_hits += 1

    # (b) Overlap@2: trajectories with ≥4 boundaries
    overlap_sum = 0.0
    overlap_total = 0
    for recs in by_traj.values():
        if len(recs) < 4:
            continue
        overlap_total += 1
        emb_ranked = sorted(range(len(recs)), key=lambda i: recs[i].emb_var_cosine, reverse=True)
        rew_ranked = sorted(range(len(recs)), key=lambda i: recs[i].reward_var, reverse=True)
        top2_emb = set(emb_ranked[:2])
        top2_rew = set(rew_ranked[:2])
        jaccard = len(top2_emb & top2_rew) / len(top2_emb | top2_rew)
        overlap_sum += jaccard

    # (c) κ_emb: reward_var at top-1 emb_var / mean reward_var (per traj, then avg)
    kappa_ratios: list[float] = []
    for recs in by_traj.values():
        if len(recs) < 2:
            continue
        mean_rv = float(np.mean([r.reward_var for r in recs]))
        if mean_rv < 1e-12:
            continue  # skip trajectories where all boundaries have reward_var ≈ 0
        best_emb_idx = max(range(len(recs)), key=lambda i: recs[i].emb_var_cosine)
        selected_rv = recs[best_emb_idx].reward_var
        kappa_ratios.append(selected_rv / mean_rv)

    return {
        "top1_agreement": top1_hits / max(1, top1_total),
        "top1_n_trajectories": top1_total,
        "overlap_at_2": overlap_sum / max(1, overlap_total),
        "overlap_at_2_n_trajectories": overlap_total,
        "kappa_emb": float(np.mean(kappa_ratios)) if kappa_ratios else float("nan"),
        "kappa_emb_n_trajectories": len(kappa_ratios),
    }


# ── Runner ──────────────────────────────────────────────────────────


@dataclass
class EmbVarRunner:
    """Runs the embedding-variance vs reward-variance diagnostic."""

    stage2_config: Stage2Config
    oracle_config: OracleConfig

    def run(
        self,
        backend: RolloutBackend,
        trajectory_boundaries: list[tuple[Trajectory, Boundary]],
        verifier: MathVerifier,
        *,
        n_continuations: int = 4,
        n_reward_samples: int | None = None,
        lookahead_max_tokens: int | None = None,
        max_new_tokens: int = 512,
        seed: int = 0,
        n_informative_prompts: int = 0,
    ) -> EmbVarResult:
        """Run embedding-variance vs reward-variance diagnostic.

        Args:
            n_continuations: Total rollouts per boundary (used for embedding).
            n_reward_samples: How many of the n_continuations to use for
                reward variance. Defaults to n_continuations.  Set to e.g. 4
                when n_continuations=8 to simulate "8 cheap lookaheads to
                decide, 4 expensive full rollouts to measure".
            lookahead_max_tokens: Override for truncation length. Defaults to
                stage2_config.lookahead_max_new_tokens.
        """
        scorer = Stage2Scorer(self.stage2_config)
        tokenizer = backend.tokenizer
        la_max = lookahead_max_tokens if lookahead_max_tokens is not None else self.stage2_config.lookahead_max_new_tokens
        n_rew = n_reward_samples if n_reward_samples is not None else n_continuations
        records: list[EmbVarBoundaryRecord] = []

        # Batch all boundary prefixes → single vLLM call
        prefixes = [
            traj.prompt_token_ids + traj.token_ids[: b.token_position]
            for traj, b in trajectory_boundaries
        ]
        terminal_rollouts = backend.continue_from_prefixes(
            prefix_token_ids=prefixes,
            n_continuations=n_continuations,
            max_new_tokens=max_new_tokens,
            temperature=self.oracle_config.concordance_terminal_temperature,
            top_p=0.95,
            seed=seed,
        )
        total_rollouts = sum(len(rs) for rs in terminal_rollouts)

        for (traj, b), rs in zip(
            trajectory_boundaries, terminal_rollouts, strict=False
        ):
            terminal_texts = [r.response_text for r in rs]
            lookahead_texts = [
                _truncate_text_first_tokens(t, tokenizer, la_max)
                for t in terminal_texts
            ]

            # Embed ALL lookaheads for embedding variance
            emb = scorer.embed(lookahead_texts)
            evc = _pairwise_cosine_var(emb)
            evt = _trace_cov(emb)

            # Terminal rewards via verifier — use only first n_rew rollouts
            reward_texts = terminal_texts[:n_rew]
            rewards: list[float] = []
            n_correct = 0
            for text in reward_texts:
                v = verifier.score(text, traj.ground_truth_answer)
                r_val = 1.0 if v.correct else 0.0
                rewards.append(r_val)
                if v.correct:
                    n_correct += 1
            reward_arr = np.asarray(rewards, dtype=float)
            reward_var = float(np.var(reward_arr, ddof=0))
            mean_reward = float(np.mean(reward_arr))

            remaining = max(0, len(traj.token_ids) - b.token_position)
            records.append(
                EmbVarBoundaryRecord(
                    trajectory_id=traj.trajectory_id,
                    boundary_idx=b.boundary_idx,
                    token_position=b.token_position,
                    relative_position=b.token_position / max(1, traj.length),
                    remaining_tokens=remaining,
                    n_samples=len(rs),
                    emb_var_cosine=evc,
                    emb_var_trace=evt,
                    reward_var=reward_var,
                    mean_reward=mean_reward,
                    n_correct=n_correct,
                    lookahead_texts=lookahead_texts,
                    terminal_texts=terminal_texts,
                )
            )

        # ── Aggregate correlations ──────────────────────────────────
        emb_cos = [r.emb_var_cosine for r in records]
        emb_tr = [r.emb_var_trace for r in records]
        rew_var = [r.reward_var for r in records]
        n_valid = len(records)

        rho_cos = _spearman(emb_cos, rew_var)
        rho_tr = _spearman(emb_tr, rew_var)
        ci_cos = fisher_z_ci(rho_cos, n_valid) if n_valid > 4 else (float("nan"), float("nan"))
        ci_tr = fisher_z_ci(rho_tr, n_valid) if n_valid > 4 else (float("nan"), float("nan"))

        # Long-only subset: remaining > 2 × la_max
        long_mask = [r.remaining_tokens > 2 * la_max for r in records]
        long_cos = [c for c, m in zip(emb_cos, long_mask) if m]
        long_tr = [c for c, m in zip(emb_tr, long_mask) if m]
        long_rew = [r for r, m in zip(rew_var, long_mask) if m]
        rho_cos_long = _spearman(long_cos, long_rew) if len(long_cos) >= 4 else float("nan")
        rho_tr_long = _spearman(long_tr, long_rew) if len(long_tr) >= 4 else float("nan")
        n_long = sum(long_mask)

        mean_remaining = float(np.mean([r.remaining_tokens for r in records])) if records else 0.0

        # ── Selection metrics (per-trajectory) ─────────────────────
        sel = _compute_selection_metrics(records)

        return EmbVarResult(
            records=records,
            top1_agreement=sel["top1_agreement"],
            top1_n_trajectories=sel["top1_n_trajectories"],
            overlap_at_2=sel["overlap_at_2"],
            overlap_at_2_n_trajectories=sel["overlap_at_2_n_trajectories"],
            kappa_emb=sel["kappa_emb"],
            kappa_emb_n_trajectories=sel["kappa_emb_n_trajectories"],
            rho_cosine=rho_cos,
            rho_trace=rho_tr,
            rho_cosine_ci=ci_cos,
            rho_trace_ci=ci_tr,
            rho_cosine_long_only=rho_cos_long,
            rho_trace_long_only=rho_tr_long,
            n_long_boundaries=n_long,
            n_informative_prompts=n_informative_prompts,
            n_boundaries=len(records),
            total_rollouts=total_rollouts,
            mean_remaining_tokens=mean_remaining,
            lookahead_max_tokens=la_max,
            n_emb_samples=n_continuations,
            n_reward_samples=n_rew,
        )


# ── Convenience function ───────────────────────────────────────────


def run_embedding_variance_check(
    backend: RolloutBackend,
    trajectory_boundaries: list[tuple[Trajectory, Boundary]],
    verifier: MathVerifier,
    oracle_config: OracleConfig,
    stage2_config: Stage2Config,
    *,
    n_continuations: int = 4,
    n_reward_samples: int | None = None,
    lookahead_max_tokens: int | None = None,
    max_new_tokens: int = 512,
    seed: int = 0,
    n_informative_prompts: int = 0,
) -> EmbVarResult:
    runner = EmbVarRunner(oracle_config=oracle_config, stage2_config=stage2_config)
    return runner.run(
        backend,
        trajectory_boundaries,
        verifier,
        n_continuations=n_continuations,
        n_reward_samples=n_reward_samples,
        lookahead_max_tokens=lookahead_max_tokens,
        max_new_tokens=max_new_tokens,
        seed=seed,
        n_informative_prompts=n_informative_prompts,
    )
