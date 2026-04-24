from __future__ import annotations

import numpy as np

from grpocredit.common.types import ForcedActionResult, OracleRecord
from grpocredit.oracle.kappa_estimator import estimate_kappa


def _mk_record(s2: float, var_q: float, h_token: float = 0.5) -> OracleRecord:
    # Fake forced actions with uniform pi so the fisher trace proxy is nontrivial
    fr = [
        ForcedActionResult(
            boundary_id="b", first_token_id=i, first_token_str=str(i),
            pi_first_token=0.25, rewards=[1.0, 0.0], response_texts=["", ""],
        )
        for i in range(4)
    ]
    return OracleRecord(
        boundary_id="b",
        trajectory_id="t",
        boundary_idx=0,
        token_position=10,
        relative_position=0.5,
        trajectory_length=20,
        coverage_c=1.0,
        forced_action_results=fr,
        h_token=h_token,
        s2=s2,
        var_q_pi=var_q,
    )


def test_kappa_is_one_when_selection_is_uninformative() -> None:
    rng = np.random.default_rng(0)
    records = [_mk_record(s2=rng.random(), var_q=rng.random()) for _ in range(200)]
    res = estimate_kappa(records, selection_score="s2", f_sel=0.15, seed=0)
    assert 0.6 <= res.kappa <= 1.6  # uninformative selection should center near 1


def test_kappa_is_large_when_s2_tracks_var() -> None:
    records = [
        _mk_record(s2=i * 0.01, var_q=(i * 0.01) ** 2)  # s2 correlated with var
        for i in range(200)
    ]
    res = estimate_kappa(records, selection_score="s2", f_sel=0.15, seed=0)
    assert res.kappa > 2.0


def test_rho_gate_formula() -> None:
    # ρ_gate = sqrt(f_target / (f_sel * κ))
    records = [_mk_record(s2=i * 0.01, var_q=(i * 0.01) ** 2) for i in range(200)]
    res = estimate_kappa(records, selection_score="s2", f_sel=0.15, f_target=0.10, seed=0)
    import math

    expected = math.sqrt(0.10 / (0.15 * max(res.kappa, 1e-9)))
    assert abs(res.rho_gate - expected) < 1e-5
