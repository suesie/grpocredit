from __future__ import annotations

import math

from grpocredit.common.config import Stage1Config
from grpocredit.common.types import Boundary, Trajectory
from grpocredit.voi.stage1_entropy import (
    Stage1Scorer,
    collision_complement,
    token_entropy,
)


def test_token_entropy_uniform_over_k() -> None:
    # Uniform over 4 tokens → H = log(4)
    lps = [math.log(0.25)] * 4
    h = token_entropy(lps)
    assert abs(h - math.log(4)) < 1e-6


def test_collision_complement_uniform_over_k() -> None:
    lps = [math.log(0.25)] * 4
    cc = collision_complement(lps)
    # ‖π‖² = 4 * (0.25)² = 0.25; complement = 0.75
    assert abs(cc - 0.75) < 1e-6


def test_w_pos_tent_peaks_at_middle() -> None:
    cfg = Stage1Config(w_pos_shape="tent")
    s = Stage1Scorer(cfg)
    T = 100
    ws = [s.w_pos(t, T) for t in range(T + 1)]
    assert max(ws) == ws[50]
    assert ws[0] == 0.0
    assert ws[T] == 0.0


def test_s1_score_is_h_times_wpos() -> None:
    cfg = Stage1Config(w_pos_shape="uniform")
    s = Stage1Scorer(cfg)
    traj = Trajectory(
        trajectory_id="t",
        prompt_id="p",
        prompt="",
        prompt_token_ids=[],
        response_text="",
        token_ids=list(range(20)),
        logprobs=[0.0] * 20,
        token_entropies=[1.2, 0.3, 2.0] + [0.5] * 17,
        reward=0.0,
        correct=False,
        ground_truth_answer="",
    )
    bds = [
        Boundary(trajectory_id="t", boundary_idx=0, token_position=0, char_position=0, kind="x"),
        Boundary(trajectory_id="t", boundary_idx=1, token_position=2, char_position=2, kind="x"),
    ]
    s.score(traj, bds)
    assert bds[0].s1 == 1.2
    assert bds[1].s1 == 2.0


def test_filter_top_keeps_highest_s1() -> None:
    cfg = Stage1Config(w_pos_shape="uniform", keep_top_pct=0.5)
    s = Stage1Scorer(cfg)
    bds = [
        Boundary(
            trajectory_id="t",
            boundary_idx=i,
            token_position=i,
            char_position=i,
            kind="x",
            s1=i * 1.0,
        )
        for i in range(4)
    ]
    kept = s.filter_top(bds)
    assert len(kept) == 2
    assert all(b.s1 >= 2 for b in kept)


def _make_traj(token_entropies: list[float]) -> Trajectory:
    n = len(token_entropies)
    return Trajectory(
        trajectory_id="t",
        prompt_id="p",
        prompt="",
        prompt_token_ids=[],
        response_text="",
        token_ids=list(range(n)),
        logprobs=[0.0] * n,
        token_entropies=token_entropies,
        reward=0.0,
        correct=False,
        ground_truth_answer="",
    )


def test_h_fwd_max_returns_max_of_window() -> None:
    cfg = Stage1Config(w_pos_shape="uniform")
    s = Stage1Scorer(cfg)
    # Window of 5 tokens starting at position 2: [0.1, 0.5, 0.2, 0.8, 0.3]
    ents = [0.0, 0.0, 0.1, 0.5, 0.2, 0.8, 0.3, 0.0, 0.0, 0.0]
    traj = _make_traj(ents)
    assert s.h_fwd_max_at(traj, 2, 5) == 0.8
    assert abs(s.h_fwd_at(traj, 2, 5) - (0.1 + 0.5 + 0.2 + 0.8 + 0.3) / 5) < 1e-9


def test_h_fwd_max_empty_trajectory() -> None:
    cfg = Stage1Config(w_pos_shape="uniform")
    s = Stage1Scorer(cfg)
    traj = _make_traj([])
    assert s.h_fwd_max_at(traj, 0, 5) == 0.0
    assert s.h_fwd_at(traj, 0, 5) == 0.0


def test_h_fwd_max_near_end_of_trajectory() -> None:
    """Window clips to available tokens when fewer than K remain."""
    cfg = Stage1Config(w_pos_shape="uniform")
    s = Stage1Scorer(cfg)
    ents = [0.1, 0.2, 0.9, 0.3]  # 4 tokens total
    traj = _make_traj(ents)
    # Position 2 with K=5 → only 2 values available: [0.9, 0.3]
    assert s.h_fwd_max_at(traj, 2, 5) == 0.9
    assert abs(s.h_fwd_at(traj, 2, 5) - (0.9 + 0.3) / 2) < 1e-9


def test_score_auto_scales_k() -> None:
    """Auto-scale: effective_k = min(h_fwd_k, max(1, remaining // 3))."""
    cfg = Stage1Config(w_pos_shape="uniform")
    s = Stage1Scorer(cfg)
    # 30-token trajectory, boundary at position 24 → remaining=6, effective_k=min(10, 2)=2
    ents = [0.1] * 24 + [0.5, 0.9, 0.2, 0.3, 0.1, 0.1]
    traj = _make_traj(ents)
    b = Boundary(trajectory_id="t", boundary_idx=0, token_position=24, char_position=0, kind="x")
    s.score(traj, [b], h_fwd_k=10)
    # effective_k = min(10, max(1, 6 // 3)) = min(10, 2) = 2
    # h_fwd = mean([0.5, 0.9]) = 0.7
    # h_fwd_max = max([0.5, 0.9]) = 0.9
    assert abs(b.h_fwd - 0.7) < 1e-9
    assert b.h_fwd_max == 0.9


def test_score_sets_h_fwd_and_h_fwd_max() -> None:
    """score() populates both h_fwd and h_fwd_max when h_fwd_k > 0."""
    cfg = Stage1Config(w_pos_shape="uniform")
    s = Stage1Scorer(cfg)
    ents = [0.1, 0.3, 0.8, 0.2, 0.1] + [0.05] * 45  # 50 tokens
    traj = _make_traj(ents)
    b = Boundary(trajectory_id="t", boundary_idx=0, token_position=1, char_position=0, kind="x")
    s.score(traj, [b], h_fwd_k=5)
    # remaining=49, effective_k = min(5, max(1, 49//3)) = min(5, 16) = 5
    # window = ents[1:6] = [0.3, 0.8, 0.2, 0.1, 0.05]
    assert b.h_fwd is not None
    assert b.h_fwd_max is not None
    assert b.h_fwd_max == 0.8
    assert abs(b.h_fwd - (0.3 + 0.8 + 0.2 + 0.1 + 0.05) / 5) < 1e-9
