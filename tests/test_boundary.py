"""Boundary-detector regex sanity: finds paragraph / step / logical markers,
respects min-separation, and applies drift fallback when needed."""

from __future__ import annotations

from grpocredit.common.config import BoundaryConfig
from grpocredit.rollout.boundary import detect_boundaries


def _uniform_offsets(text: str, approx_tokens: int | None = None) -> tuple[list[tuple[int, int]], int]:
    """Build a synthetic (token_id → char span) mapping by splitting on whitespace.

    Good enough for boundary tests — we only need token positions to roughly
    align with the regex matches.
    """
    tokens = text.split(" ")
    if approx_tokens is not None and approx_tokens != len(tokens):
        # simulate BPE subwords by further splitting every 5th token
        new_tokens: list[str] = []
        for i, t in enumerate(tokens):
            if i % 5 == 0 and len(t) > 2:
                new_tokens.extend([t[: len(t) // 2], t[len(t) // 2 :]])
            else:
                new_tokens.append(t)
        tokens = new_tokens
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for t in tokens:
        offsets.append((cursor, cursor + len(t)))
        cursor += len(t) + 1  # +1 for the space
    return offsets, len(tokens)


def test_finds_paragraph_and_step_markers() -> None:
    text = (
        "Step 1: Start by multiplying both sides.\n\n"
        "Therefore the LHS becomes 2x.\n\n"
        "Step 2: Solve for x.\n\n"
        "Thus x = 5 and we are done."
    )
    offsets, n = _uniform_offsets(text)
    bds = detect_boundaries(
        trajectory_id="t0",
        response_text=text,
        offset_mapping=offsets,
        num_tokens=n,
    )
    kinds = [b.kind for b in bds]
    assert "paragraph" in kinds
    assert "step" in kinds
    assert "therefore" in kinds or "thus" in kinds


def test_min_separation_dedups_close_markers() -> None:
    text = "Therefore\n\nHowever the result holds."
    offsets, n = _uniform_offsets(text)
    cfg = BoundaryConfig(min_tokens_between_boundaries=4)
    bds = detect_boundaries(
        trajectory_id="t0",
        response_text=text,
        offset_mapping=offsets,
        num_tokens=n,
        config=cfg,
    )
    positions = [b.token_position for b in bds]
    for a, b in zip(positions, positions[1:], strict=False):
        assert b - a >= cfg.min_tokens_between_boundaries


def test_drift_fallback_triggers_when_few_boundaries() -> None:
    # No paragraph breaks, no step markers — should trigger sentence-punct and/or byte fallback.
    text = (
        "First we compute A. Then we compute B. Then we compute C. "
        "This gives us D. The final answer is 42."
    )
    offsets, n = _uniform_offsets(text)
    cfg = BoundaryConfig(
        min_boundaries_for_drift_fallback=3,
        max_boundaries_per_trajectory=10,
        byte_budget_stride=4,
        min_tokens_between_boundaries=1,
    )
    bds = detect_boundaries(
        trajectory_id="t0",
        response_text=text,
        offset_mapping=offsets,
        num_tokens=n,
        config=cfg,
    )
    kinds = {b.kind for b in bds}
    assert "punct" in kinds or "byte" in kinds
    assert len(bds) >= cfg.min_boundaries_for_drift_fallback


def test_respects_max_boundaries_cap() -> None:
    # Pathologically punctuation-heavy text
    text = ". ".join(f"sent{i}" for i in range(200))
    offsets, n = _uniform_offsets(text)
    cfg = BoundaryConfig(max_boundaries_per_trajectory=5, min_tokens_between_boundaries=1)
    bds = detect_boundaries(
        trajectory_id="t0",
        response_text=text,
        offset_mapping=offsets,
        num_tokens=n,
        config=cfg,
    )
    assert len(bds) <= cfg.max_boundaries_per_trajectory
