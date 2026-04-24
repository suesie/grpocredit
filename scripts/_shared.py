"""Helpers shared across sprint_d*.py scripts.

Single source of truth for:
- turning a list of `RolloutResult` into `Trajectory` objects with reward/correctness,
- running boundary detection over a batch of trajectories,
- extracting offset_mapping robustly from whatever tokenizer we're using.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from grpocredit.common.config import BoundaryConfig
from grpocredit.common.types import Boundary, RolloutResult, Trajectory
from grpocredit.rollout.boundary import BoundaryDetector
from grpocredit.rollout.datasets import PromptRecord, format_prompt
from grpocredit.rollout.verifier import MathVerifier

log = logging.getLogger(__name__)


def build_prompts(
    records: list[PromptRecord], tokenizer: Any, template: str = "math_instruct"
) -> list[str]:
    return [format_prompt(r.question, tokenizer, template=template) for r in records]


def offset_mapping_from_tokenizer(
    tokenizer: Any, text: str, token_ids: list[int]
) -> list[tuple[int, int]]:
    """Return (start, end) char offsets per *generated* token.

    Path 1: re-encode `text` with `return_offsets_mapping=True` when the
    tokenizer is a `PreTrainedTokenizerFast`.
    Path 2: fall back to greedy decode — detokenize one id at a time and
    track the cursor. Less accurate for subword tokenizers with whitespace
    merging, but works for any tokenizer.
    """
    try:
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc.get("offset_mapping") or []
        if len(offsets) >= len(token_ids):
            return [tuple(x) for x in offsets[: len(token_ids)]]
    except (TypeError, KeyError, AttributeError):
        pass
    # Fallback greedy decoding
    offsets: list[tuple[int, int]] = []
    cursor = 0
    buf = ""
    for tid in token_ids:
        prev = buf
        buf = tokenizer.decode(token_ids[: len(offsets) + 1], skip_special_tokens=True)
        new = buf[len(prev):] if buf.startswith(prev) else buf
        end = cursor + len(new)
        offsets.append((cursor, end))
        cursor = end
    return offsets


def results_to_trajectories(
    records: list[PromptRecord],
    rollouts: list[list[RolloutResult]],
    verifier: MathVerifier,
) -> list[Trajectory]:
    trajectories: list[Trajectory] = []
    for pr, group in zip(records, rollouts, strict=False):
        for g_idx, rr in enumerate(group):
            v = verifier.score(rr.response_text, pr.ground_truth_answer)
            traj = Trajectory(
                trajectory_id=f"{pr.prompt_id}#{g_idx}",
                prompt_id=pr.prompt_id,
                prompt=rr.prompt,
                prompt_token_ids=rr.prompt_token_ids,
                response_text=rr.response_text,
                token_ids=rr.token_ids,
                logprobs=rr.logprobs,
                token_entropies=rr.token_entropies,
                reward=1.0 if v.correct else 0.0,
                correct=v.correct,
                ground_truth_answer=pr.ground_truth_answer,
                extracted_answer=v.extracted,
                finish_reason=rr.finish_reason,
                group_id=pr.prompt_id,
                extra={"verifier_method": v.method, "source": pr.source},
            )
            trajectories.append(traj)
    return trajectories


def detect_all_boundaries(
    trajectories: list[Trajectory],
    tokenizer: Any,
    boundary_cfg: BoundaryConfig,
) -> dict[str, list[Boundary]]:
    detector = BoundaryDetector(boundary_cfg)
    out: dict[str, list[Boundary]] = {}
    for t in trajectories:
        offsets = offset_mapping_from_tokenizer(
            tokenizer, t.response_text, t.token_ids
        )
        bds = detector.detect(
            trajectory_id=t.trajectory_id,
            response_text=t.response_text,
            offset_mapping=offsets,
            num_tokens=len(t.token_ids),
        )
        out[t.trajectory_id] = bds
    return out


def ensure_paths(output_dir: str | Path) -> Path:
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p
