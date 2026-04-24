from __future__ import annotations

from grpocredit.advantage.shrinkage import (
    apply_shrinkage,
    james_stein_alpha,
    se_shrinkage,
)


def test_james_stein_monotone_in_M() -> None:
    alphas = [james_stein_alpha(m) for m in [1, 2, 4, 8, 16, 32]]
    assert all(a < b for a, b in zip(alphas, alphas[1:], strict=False))
    assert alphas[0] < 0.5
    assert alphas[-1] > 0.85


def test_se_shrinkage_decreases_with_noise() -> None:
    a = se_shrinkage(1.0, var_v=0.0, m=10)
    b = se_shrinkage(1.0, var_v=1.0, m=10)
    c = se_shrinkage(1.0, var_v=10.0, m=10)
    assert a > b > c


def test_apply_shrinkage_dispatch() -> None:
    assert apply_shrinkage(1.0, m=1, mode="none") == 1.0
    assert 0 < apply_shrinkage(1.0, m=1, mode="james_stein") < 1.0
    assert 0 < apply_shrinkage(1.0, m=1, mode="se", var_v=1.0) < 1.0
