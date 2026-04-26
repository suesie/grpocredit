"""Syntactic boundary detection + post-RL drift fallback.

Match boundaries in *text space* (fast regex over decoded response), then
convert to *token space* using cumulative token→char offsets that the
tokenizer exposes via `offset_mapping`. This avoids re-tokenising every
candidate boundary and keeps boundary positions aligned with the exact
token-level prefix that the vLLM runner will force.

Design notes:
- Each `Boundary` carries a `kind` tag (paragraph / step / therefore / ... /
  punct / byte). Analysis scripts use this to stratify selector survival
  by boundary type.
- If the number of syntactic boundaries falls below
  `boundary_cfg.min_boundaries_for_drift_fallback`, we *extend* the candidate
  set with sentence-punctuation boundaries, then with every-N-tokens boundaries
  — both are still independent of rollout rewards, so the §3.1.2 independence
  condition is preserved.
- `min_tokens_between_boundaries` dedups boundaries that collapse to the
  same segment (e.g., paragraph break immediately followed by 'Therefore').
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from grpocredit.common.config import BoundaryConfig
from grpocredit.common.types import Boundary

log = logging.getLogger(__name__)


@dataclass
class _RawBoundary:
    char_position: int
    kind: str
    marker_text: str


def _find_all_matches(text: str, config: BoundaryConfig) -> list[_RawBoundary]:
    matches: list[_RawBoundary] = []

    for m in re.finditer(config.paragraph_break, text):
        matches.append(_RawBoundary(m.end(), "paragraph", m.group(0)))

    for m in re.finditer(config.step_marker, text):
        matches.append(_RawBoundary(m.start(), "step", m.group(0)))

    for pattern in config.logical_markers:
        kind = _classify_marker(pattern)
        for m in re.finditer(pattern, text):
            matches.append(_RawBoundary(m.start(), kind, m.group(0)))

    matches.sort(key=lambda x: x.char_position)
    return matches


def _classify_marker(pattern: str) -> str:
    # Extract the first lowercase word from the regex for a readable `kind` tag.
    m = re.search(r"\\b([A-Za-z]+)", pattern)
    if m:
        return m.group(1).lower()
    if "boxed" in pattern:
        return "boxed"
    return "marker"


def _char_to_token(char_pos: int, offset_mapping: list[tuple[int, int]]) -> int:
    """Given a char index into the decoded response, find the token index that
    starts at or after char_pos. Returns the index into the *generated* tokens.
    """
    for i, (start, _end) in enumerate(offset_mapping):
        if start >= char_pos:
            return i
    return len(offset_mapping)


# Boundary-kind priority for kind-aware dedup. When two boundaries fall within
# `min_tokens_between_boundaries`, two rules cooperate:
#   1. A higher-priority kind escapes dedup even if it is closer than min_dist
#      to the previously kept boundary (it carries strictly more information).
#   2. If two boundaries land at the *same* token position, the higher-priority
#      kind is the one we surface for that position.
# Logical connectives (`Therefore`, `Thus`, `Hence`, …) carry the most
# step-by-step reasoning signal, so they outrank generic step markers and
# paragraph breaks; the boxed-answer marker is the highest-value boundary in
# math problems.
_KIND_PRIORITY: dict[str, int] = {
    "boxed": 6,
    "therefore": 5,
    "hence": 5,
    "thus": 5,
    "step": 4,
    "however": 3,
    "but": 3,
    "so": 3,
    "paragraph": 2,
    "punct": 1,
    "byte": 0,
}


def _kind_priority(kind: str) -> int:
    return _KIND_PRIORITY.get(kind, 2)


def _dedup_by_token_distance(
    boundaries: list[Boundary], min_dist: int
) -> list[Boundary]:
    if not boundaries:
        return []
    kept: list[Boundary] = [boundaries[0]]
    for b in boundaries[1:]:
        gap = b.token_position - kept[-1].token_position
        if gap >= min_dist:
            kept.append(b)
            continue
        # Within the dedup window. If the new boundary's kind is *strictly*
        # higher priority than the last kept, surface it as well — a paragraph
        # break immediately followed by `Therefore` should yield both segments
        # of credit-assignment information, not just the paragraph break.
        if _kind_priority(b.kind) > _kind_priority(kept[-1].kind):
            kept.append(b)
    return kept


@dataclass
class BoundaryDetector:
    """Applies the §3.2.2 syntactic delimiter rules with §3.2.2 drift fallback."""

    config: BoundaryConfig

    def detect(
        self,
        *,
        trajectory_id: str,
        response_text: str,
        offset_mapping: list[tuple[int, int]],
        num_tokens: int,
    ) -> list[Boundary]:
        """`offset_mapping[i] = (start_char, end_char)` for generated token `i`."""
        if num_tokens <= 1:
            return []
        if len(offset_mapping) != num_tokens:
            log.warning(
                "BoundaryDetector: offset_mapping length %d != num_tokens %d; "
                "truncating to min",
                len(offset_mapping),
                num_tokens,
            )
            n = min(len(offset_mapping), num_tokens)
            offset_mapping = offset_mapping[:n]
            num_tokens = n

        raw = _find_all_matches(response_text, self.config)
        boundaries: list[Boundary] = []
        for idx, rm in enumerate(raw):
            tp = _char_to_token(rm.char_position, offset_mapping)
            # Skip degenerate boundaries at the very end. Position 0 is only
            # degenerate for paragraph breaks (they mark a *prior* break that
            # doesn't exist before the response starts); informative markers
            # like "Step 1:" or "\\boxed{" can legitimately sit at position 0.
            if tp >= num_tokens - 1:
                continue
            if tp < 0:
                continue
            if tp == 0 and rm.kind == "paragraph":
                continue
            boundaries.append(
                Boundary(
                    trajectory_id=trajectory_id,
                    boundary_idx=idx,
                    token_position=tp,
                    char_position=rm.char_position,
                    kind=rm.kind,
                    marker_text=rm.marker_text,
                )
            )

        boundaries.sort(key=lambda b: b.token_position)
        boundaries = _dedup_by_token_distance(
            boundaries, self.config.min_tokens_between_boundaries
        )

        # Drift fallback: if too few syntactic boundaries, augment.
        if len(boundaries) < self.config.min_boundaries_for_drift_fallback:
            boundaries = self._apply_drift_fallback(
                trajectory_id, response_text, offset_mapping, num_tokens, boundaries
            )

        # Cap to avoid blowing out compute at low-quality boundaries.
        if len(boundaries) > self.config.max_boundaries_per_trajectory:
            stride = max(1, len(boundaries) // self.config.max_boundaries_per_trajectory)
            boundaries = boundaries[::stride][
                : self.config.max_boundaries_per_trajectory
            ]

        # Re-number boundary_idx sequentially after dedup/subsampling
        for i, b in enumerate(boundaries):
            b.boundary_idx = i

        return boundaries

    def _apply_drift_fallback(
        self,
        trajectory_id: str,
        response_text: str,
        offset_mapping: list[tuple[int, int]],
        num_tokens: int,
        existing: list[Boundary],
    ) -> list[Boundary]:
        # Layer 1: sentence punctuation
        extra: list[Boundary] = list(existing)
        for m in re.finditer(self.config.sentence_punct, response_text):
            tp = _char_to_token(m.end(), offset_mapping)
            if 0 < tp < num_tokens - 1:
                extra.append(
                    Boundary(
                        trajectory_id=trajectory_id,
                        boundary_idx=-1,
                        token_position=tp,
                        char_position=m.end(),
                        kind="punct",
                        marker_text=m.group(0),
                    )
                )
        extra.sort(key=lambda b: b.token_position)
        extra = _dedup_by_token_distance(
            extra, self.config.min_tokens_between_boundaries
        )

        # Layer 2: every-N-tokens byte budget
        if len(extra) < self.config.min_boundaries_for_drift_fallback:
            stride = self.config.byte_budget_stride
            for tp in range(stride, num_tokens - 1, stride):
                char_pos = offset_mapping[tp][0]
                extra.append(
                    Boundary(
                        trajectory_id=trajectory_id,
                        boundary_idx=-1,
                        token_position=tp,
                        char_position=char_pos,
                        kind="byte",
                        marker_text=f"<byte@{tp}>",
                    )
                )
            extra.sort(key=lambda b: b.token_position)
            extra = _dedup_by_token_distance(
                extra, self.config.min_tokens_between_boundaries
            )
        return extra


# Convenience functional alias
def detect_boundaries(
    *,
    trajectory_id: str,
    response_text: str,
    offset_mapping: list[tuple[int, int]],
    num_tokens: int,
    config: BoundaryConfig | None = None,
) -> list[Boundary]:
    cfg = config or BoundaryConfig()
    return BoundaryDetector(cfg).detect(
        trajectory_id=trajectory_id,
        response_text=response_text,
        offset_mapping=offset_mapping,
        num_tokens=num_tokens,
    )
