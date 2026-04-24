"""Concordance check (RC-F1, plan §2A).

At each boundary, sample K_LA=4 *long* continuations that terminate at the
end-of-sequence. For each continuation:
  - Take its first ≤30 tokens as the *lookahead* L_i.
  - Take the full continuation as the *terminal* T_i.

Cluster {L_1..L_K} and {T_1..T_K} separately via sentence-T5 cosine ≥ 0.85.
Per boundary, compute MI(C_prefix, C_terminal) using the Hungarian
alignment — we aggregate per-boundary MI and report the mean (+ per-position
deciles).

Gate thresholds (plan §2A):
    > 0.3 bits → proceed with Stage 2 as designed.
    0.15–0.3 bits → upgrade to NLI clustering (3% cost) and re-check.
    ≤ 0.15 bits → Stage 2 is noise; pivot to Plan B (§8 of plan).

Implementation notes:
- Per-boundary MI is the right aggregation — cluster labels "1" at boundary
  b₁ are unrelated to labels "1" at b₂, so MI-over-concatenated-pairs would
  double-count coincidental label overlaps.
- With K_LA=4 per boundary the per-boundary MI estimate is coarse;
  confidence comes from aggregation over ≥500 boundaries.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from grpocredit.common.config import OracleConfig, Stage2Config
from grpocredit.common.types import Boundary, Trajectory
from grpocredit.rollout.vllm_runner import RolloutBackend
from grpocredit.voi.stage2_semantic import (
    Stage2Scorer,
    cluster_sizes_from_labels,
    connected_component_clusters,
)

log = logging.getLogger(__name__)


@dataclass
class ConcordanceBoundaryRecord:
    trajectory_id: str
    boundary_idx: int
    token_position: int
    relative_position: float
    n_samples: int
    prefix_clusters: int
    terminal_clusters: int
    mi_bits: float  # per-boundary MI estimate in bits
    lookahead_texts: list[str] = field(default_factory=list)
    terminal_texts: list[str] = field(default_factory=list)
    prefix_labels: list[int] = field(default_factory=list)
    terminal_labels: list[int] = field(default_factory=list)


@dataclass
class ConcordanceResult:
    mean_mi_bits: float
    median_mi_bits: float
    mi_by_position_decile: list[tuple[int, float, int]]  # (decile, mean_mi, n_boundaries)
    per_boundary: list[ConcordanceBoundaryRecord]
    total_rollouts: int
    config: OracleConfig
    clustering_method: str = "cosine"  # or 'nli'


def _mi_bits(labels_x: list[int], labels_y: list[int]) -> float:
    """Empirical plug-in MI in bits for a small joint table."""
    if not labels_x or not labels_y:
        return 0.0
    n = len(labels_x)
    from collections import Counter

    cx = Counter(labels_x)
    cy = Counter(labels_y)
    joint: Counter[tuple[int, int]] = Counter(zip(labels_x, labels_y, strict=False))
    mi = 0.0
    for (x, y), c_xy in joint.items():
        p_xy = c_xy / n
        p_x = cx[x] / n
        p_y = cy[y] / n
        if p_xy > 0 and p_x > 0 and p_y > 0:
            mi += p_xy * math.log2(p_xy / (p_x * p_y))
    return mi


def _truncate_text_first_tokens(
    text: str, tokenizer: Any, max_tokens: int
) -> str:
    """Decode the first `max_tokens` tokens of an already-generated response.

    Used to derive L_i from a terminal rollout without a second generation call.
    """
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)


@dataclass
class ConcordanceRunner:
    """Single-purpose runner: full-terminal rollouts → pair (L_i, T_i) → MI."""

    oracle_config: OracleConfig
    stage2_config: Stage2Config

    def run(
        self,
        backend: RolloutBackend,
        trajectory_boundaries: list[tuple[Trajectory, Boundary]],
        *,
        max_new_tokens: int,
        seed: int = 0,
    ) -> ConcordanceResult:
        """Execute concordance across many boundaries and report aggregate MI."""
        scorer = Stage2Scorer(self.stage2_config)
        tokenizer = backend.tokenizer
        records: list[ConcordanceBoundaryRecord] = []
        total_rollouts = 0

        # Batch all boundaries to vLLM in one `continue_from_prefixes` call
        # (vLLM handles the batching — 1 call is much faster than many).
        prefixes = [
            traj.prompt_token_ids + traj.token_ids[: b.token_position]
            for traj, b in trajectory_boundaries
        ]
        terminal_rollouts = backend.continue_from_prefixes(
            prefix_token_ids=prefixes,
            n_continuations=self.oracle_config.concordance_lookaheads,
            max_new_tokens=max_new_tokens,
            temperature=self.oracle_config.concordance_terminal_temperature,
            top_p=0.95,
            seed=seed,
        )
        total_rollouts = sum(len(rs) for rs in terminal_rollouts)

        # Cluster each boundary's (L, T) pair.
        for (traj, b), rs in zip(trajectory_boundaries, terminal_rollouts, strict=False):
            terminal_texts = [r.response_text for r in rs]
            lookahead_texts = [
                _truncate_text_first_tokens(
                    t,
                    tokenizer,
                    self.stage2_config.lookahead_max_new_tokens,
                )
                for t in terminal_texts
            ]

            # Embed both side-by-side so the encoder is loaded once per batch.
            all_texts = lookahead_texts + terminal_texts
            emb = scorer.embed(all_texts)
            n = len(lookahead_texts)
            emb_l = emb[:n]
            emb_t = emb[n:]

            labels_l = connected_component_clusters(
                emb_l, self.stage2_config.cosine_threshold
            )
            labels_t = connected_component_clusters(
                emb_t, self.stage2_config.cosine_threshold
            )
            mi = _mi_bits(labels_l, labels_t)

            sizes_l = cluster_sizes_from_labels(labels_l)
            sizes_t = cluster_sizes_from_labels(labels_t)

            records.append(
                ConcordanceBoundaryRecord(
                    trajectory_id=traj.trajectory_id,
                    boundary_idx=b.boundary_idx,
                    token_position=b.token_position,
                    relative_position=b.token_position / max(1, traj.length),
                    n_samples=n,
                    prefix_clusters=len(sizes_l),
                    terminal_clusters=len(sizes_t),
                    mi_bits=mi,
                    lookahead_texts=lookahead_texts,
                    terminal_texts=terminal_texts,
                    prefix_labels=labels_l,
                    terminal_labels=labels_t,
                )
            )

        mi_values = [r.mi_bits for r in records]
        mean_mi = float(np.mean(mi_values)) if mi_values else 0.0
        median_mi = float(np.median(mi_values)) if mi_values else 0.0

        # Per-position-decile breakdown
        by_decile: list[tuple[int, float, int]] = []
        if records:
            deciles = np.digitize(
                [r.relative_position for r in records], np.linspace(0, 1, 11)
            ) - 1
            for d in range(10):
                mask = deciles == d
                vals = [records[i].mi_bits for i in range(len(records)) if mask[i]]
                by_decile.append((d, float(np.mean(vals)) if vals else 0.0, len(vals)))

        return ConcordanceResult(
            mean_mi_bits=mean_mi,
            median_mi_bits=median_mi,
            mi_by_position_decile=by_decile,
            per_boundary=records,
            total_rollouts=total_rollouts,
            config=self.oracle_config,
            clustering_method="cosine",
        )


def run_concordance_check(
    backend: RolloutBackend,
    trajectory_boundaries: list[tuple[Trajectory, Boundary]],
    oracle_config: OracleConfig,
    stage2_config: Stage2Config,
    *,
    max_new_tokens: int,
    seed: int = 0,
) -> ConcordanceResult:
    runner = ConcordanceRunner(oracle_config=oracle_config, stage2_config=stage2_config)
    return runner.run(
        backend,
        trajectory_boundaries,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )
