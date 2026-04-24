from __future__ import annotations

import numpy as np

from grpocredit.common.types import OracleRecord
from grpocredit.oracle.position_curve import (
    _classify_shape,
    compute_position_curve,
)


def _mk_record(rel_pos: float, var: float) -> OracleRecord:
    return OracleRecord(
        boundary_id="b",
        trajectory_id="t",
        boundary_idx=0,
        token_position=int(rel_pos * 100),
        relative_position=rel_pos,
        trajectory_length=100,
        coverage_c=1.0,
        forced_action_results=[],
        var_q_pi=var,
    )


def test_mid_peak_classification() -> None:
    # Var curve peaks at decile 5 (mid-trajectory)
    rng = np.random.default_rng(0)
    records: list[OracleRecord] = []
    for _ in range(500):
        rel = rng.random()
        var = np.exp(-((rel - 0.5) / 0.15) ** 2)
        records.append(_mk_record(rel, var))
    curve = compute_position_curve(records, n_bins=10)
    assert curve.shape_classification == "mid_peak"
    assert 3 <= curve.peak_decile <= 6


def test_flat_classification() -> None:
    rng = np.random.default_rng(1)
    records = [_mk_record(rng.random(), 0.5 + 0.01 * rng.standard_normal()) for _ in range(500)]
    curve = compute_position_curve(records, n_bins=10)
    assert curve.shape_classification == "flat"


def test_classify_shape_directly() -> None:
    # end-loaded curve
    vals = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 1.0]
    s, peak = _classify_shape(vals)
    assert s == "end_loaded"
    assert peak >= 8
