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
