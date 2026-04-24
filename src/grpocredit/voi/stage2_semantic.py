"""Stage 2 — semantic-entropy confirmation via K_LA lookaheads (§3.2.3).

Per Stage-1 survivor boundary `b`:
  1. Sample `K_LA = 4` short continuations (≤30 tokens) from s_b.
  2. Embed each via sentence-T5-base (frozen external encoder, *not* the live
     policy's hidden states — those drift during training and represent
     next-token features, not meaning).
  3. Cluster by cosine ≥ τ_c (default 0.85) → connected components on the
     similarity graph.
  4. Compute H_sem(b) from cluster-size proportions.

D2 decision: at K_LA=4 we default to a binary gate (single cluster → skip;
multi-cluster → probe) because continuous H_sem at K_LA=4 is too noisy. The
`gate_mode = 'continuous'` switch upgrades to the full H_sem = −Σ P log P — use
it only after bumping K_LA to 8.

Selection-bias lemma (§3.2.3) is respected as long as the lookahead
continuations `L_1..L_K` are *discarded* before the probe rollouts run; this
module only scores, it does not reuse the lookaheads as training signal.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from grpocredit.common.config import Stage2Config
from grpocredit.common.types import Boundary, RolloutResult, Trajectory
from grpocredit.rollout.vllm_runner import RolloutBackend

log = logging.getLogger(__name__)


def semantic_entropy(cluster_sizes: list[int]) -> float:
    total = sum(cluster_sizes)
    if total <= 1:
        return 0.0
    h = 0.0
    for s in cluster_sizes:
        if s > 0:
            p = s / total
            h -= p * math.log(p)
    return h


def connected_component_clusters(
    embeddings: np.ndarray, cosine_threshold: float
) -> list[int]:
    """Return a cluster label per row using cosine ≥ threshold as edges.

    embeddings : shape (n, d). Assumes rows are L2-normalised; if not, we
    normalise in place.
    """
    if len(embeddings) == 0:
        return []
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    e = embeddings / norms
    sim = e @ e.T

    n = len(e)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= cosine_threshold:
                union(i, j)

    root_to_label: dict[int, int] = {}
    labels = [0] * n
    next_label = 0
    for i in range(n):
        r = find(i)
        if r not in root_to_label:
            root_to_label[r] = next_label
            next_label += 1
        labels[i] = root_to_label[r]
    return labels


def cluster_sizes_from_labels(labels: list[int]) -> list[int]:
    if not labels:
        return []
    k = max(labels) + 1
    sizes = [0] * k
    for lbl in labels:
        sizes[lbl] += 1
    return sizes


@dataclass
class Stage2Scorer:
    """Orchestrates sentence-T5 clustering over K_LA lookaheads.

    Loads the encoder lazily on first use — so importing this module is free.
    """

    config: Stage2Config
    _encoder: Any = field(default=None, init=False, repr=False)
    _nli: Any = field(default=None, init=False, repr=False)

    def _encoder_model(self) -> Any:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            log.info("Stage2: loading encoder %s", self.config.encoder)
            self._encoder = SentenceTransformer(self.config.encoder)
        return self._encoder

    def _nli_model(self) -> Any:
        if self._nli is None and self.config.nli_fallback_model:
            from transformers import pipeline

            log.info("Stage2: loading NLI fallback %s", self.config.nli_fallback_model)
            self._nli = pipeline(
                "text-classification",
                model=self.config.nli_fallback_model,
                top_k=None,
            )
        return self._nli

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 768))
        enc = self._encoder_model()
        emb = enc.encode(
            texts,
            batch_size=self.config.encoder_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(emb, dtype=np.float32)

    def cluster_lookaheads(self, lookahead_texts: list[str]) -> tuple[list[int], list[int]]:
        """Return (labels, cluster_sizes) for the K_LA continuations."""
        if not lookahead_texts:
            return [], []
        emb = self.embed(lookahead_texts)
        labels = connected_component_clusters(emb, self.config.cosine_threshold)
        return labels, cluster_sizes_from_labels(labels)

    def h_sem_from_sizes(self, sizes: list[int]) -> float:
        if self.config.gate_mode == "binary":
            return 1.0 if len(sizes) > 1 else 0.0
        if self.config.gate_mode == "continuous":
            return semantic_entropy(sizes)
        raise ValueError(f"Unknown gate_mode: {self.config.gate_mode}")

    def run_lookaheads(
        self,
        backend: RolloutBackend,
        trajectory: Trajectory,
        boundaries: list[Boundary],
        *,
        seed_offset: int = 0,
    ) -> dict[int, list[RolloutResult]]:
        """Run K_LA lookaheads for each boundary. Returns {id(boundary): rollouts}.

        Keyed by Python object identity so the dict is safe to merge across
        trajectories (boundary_idx is only unique *within* one trajectory).
        """
        if not boundaries:
            return {}
        prefixes = [
            trajectory.prompt_token_ids + trajectory.token_ids[: b.token_position]
            for b in boundaries
        ]
        rollouts_per_boundary = backend.continue_from_prefixes(
            prefix_token_ids=prefixes,
            n_continuations=self.config.n_lookaheads,
            max_new_tokens=self.config.lookahead_max_new_tokens,
            temperature=self.config.lookahead_temperature,
            top_p=0.95,
            seed=seed_offset,
        )
        return {id(b): rs for b, rs in zip(boundaries, rollouts_per_boundary, strict=False)}

    def score(
        self,
        boundaries: list[Boundary],
        lookahead_rollouts: dict[int, list[RolloutResult]],
    ) -> list[Boundary]:
        """Compute H_sem and s_2 = H_token · H_sem for each boundary.

        `lookahead_rollouts` is keyed by `id(boundary)` — matching what
        `run_lookaheads` returns — so the same dict can be reused across
        trajectory batches.

        Caller is expected to have already filled `b.h_token` and `b.s1` via
        `Stage1Scorer`. The `w_pos` term in §3.2.3's composite lives in `s1`
        — we do *not* multiply by it again here.
        """
        for b in boundaries:
            rollouts = lookahead_rollouts.get(id(b), [])
            texts = [r.response_text for r in rollouts]
            _labels, sizes = self.cluster_lookaheads(texts)
            h_sem = self.h_sem_from_sizes(sizes)
            b.h_sem = h_sem
            b.s2 = (b.h_token or 0.0) * h_sem
        return boundaries

    def filter_top(
        self, boundaries: list[Boundary], keep_top_pct: float | None = None
    ) -> list[Boundary]:
        if not boundaries:
            return []
        p = keep_top_pct if keep_top_pct is not None else self.config.keep_top_pct
        if p >= 1.0:
            return list(boundaries)
        k = max(1, int(round(len(boundaries) * p)))
        sorted_b = sorted(boundaries, key=lambda b: b.s2 or 0.0, reverse=True)
        survivors = sorted_b[:k]
        survivor_ids = {id(b) for b in survivors}
        for b in boundaries:
            if id(b) not in survivor_ids and b.stage_stopped_at == 0:
                b.stage_stopped_at = 2
        return survivors
