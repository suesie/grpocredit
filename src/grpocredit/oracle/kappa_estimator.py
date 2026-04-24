"""κ — variance concentration factor (RC-F4, plan §2B).

κ = (avg per-token gradient variance at *selected* boundaries) /
    (avg per-token gradient variance across *all* boundaries)

Per-token gradient variance proxy: f_t · Var_{a∼π}(Q^π(s_b, a))
where f_t = 1 − ‖π(·|s_b)‖² is the token-level Fisher-trace.

Plan §2B: "For the top-15% of boundaries ranked by s_2: compute the ratio of
average per-token gradient variance at those boundaries to average per-token
gradient variance across all boundaries."

Also computes the ρ-gate: ρ_gate = sqrt(f_target / (f_sel · κ)) where
`f_target` = 0.10 (10% trajectory-level gradient-variance reduction target)
and `f_sel` = 0.15 (fraction of boundaries selected).

κ < 2 → selection story is too weak to headline; paper pivots to efficiency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from grpocredit.common.types import OracleRecord


@dataclass
class KappaResult:
    kappa: float
    kappa_ci_low: float
    kappa_ci_high: float
    rho_gate: float
    mean_grad_var_selected: float
    mean_grad_var_all: float
    f_sel: float
    selection_score: str  # 's2' | 's1' | 'h_sem' | 'h_token'


def _fisher_trace_proxy(h_token: float | None, probs: list[float] | None = None) -> float:
    """f_t = 1 − ‖π‖². If only H_token is available, approximate via 1 − exp(−H).

    The best signal is the actual top-k probs from the policy — but the offline
    oracle already saves forced-action `pi_first_token` entries, from which we
    can compute 1 − Σ π(a)² over the top-M.
    """
    if probs is not None:
        s = sum(p * p for p in probs)
        return max(0.0, 1.0 - s)
    if h_token is not None and h_token > 0:
        return 1.0 - math.exp(-h_token)
    return 0.0


def _grad_var_proxy(record: OracleRecord) -> float:
    """f_t · Var_{a∼π}(Q^π) using the head probabilities the oracle has."""
    if record.var_q_pi is None:
        return float("nan")
    head_probs = [fr.pi_first_token for fr in record.forced_action_results]
    f_t = _fisher_trace_proxy(record.h_token, head_probs if head_probs else None)
    return f_t * record.var_q_pi


def _bootstrap_kappa(
    gv_all: np.ndarray,
    mask_sel: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile-bootstrap 95% CI for κ."""
    rng = np.random.default_rng(seed)
    n = len(gv_all)
    if n == 0 or mask_sel.sum() == 0:
        return float("nan"), float("nan")
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        gv_b = gv_all[idx]
        mask_b = mask_sel[idx]
        denom = np.nanmean(gv_b)
        numer = np.nanmean(gv_b[mask_b]) if mask_b.sum() > 0 else np.nan
        boots[i] = numer / denom if denom > 0 else np.nan
    return float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))


def estimate_kappa(
    records: list[OracleRecord],
    *,
    selection_score: str = "s2",
    f_sel: float = 0.15,
    f_target: float = 0.10,
    n_boot: int = 1000,
    seed: int = 0,
) -> KappaResult:
    """κ + bootstrap CI + ρ_gate.

    Parameters
    ----------
    selection_score : which field of `OracleRecord` to rank boundaries by.
    f_sel : fraction selected; 0.15 matches plan §2B ("top-15%").
    f_target : target trajectory-level gradient-variance reduction (10%).
    """
    if not records:
        return KappaResult(
            kappa=float("nan"),
            kappa_ci_low=float("nan"),
            kappa_ci_high=float("nan"),
            rho_gate=float("nan"),
            mean_grad_var_selected=float("nan"),
            mean_grad_var_all=float("nan"),
            f_sel=f_sel,
            selection_score=selection_score,
        )

    gv = np.asarray([_grad_var_proxy(r) for r in records], dtype=float)
    scores = np.asarray(
        [getattr(r, selection_score) or 0.0 for r in records], dtype=float
    )
    valid = ~np.isnan(gv)
    gv = gv[valid]
    scores = scores[valid]

    if len(gv) == 0:
        return KappaResult(
            kappa=float("nan"),
            kappa_ci_low=float("nan"),
            kappa_ci_high=float("nan"),
            rho_gate=float("nan"),
            mean_grad_var_selected=float("nan"),
            mean_grad_var_all=float("nan"),
            f_sel=f_sel,
            selection_score=selection_score,
        )

    k = max(1, int(round(len(gv) * f_sel)))
    top_idx = np.argsort(-scores)[:k]
    mask = np.zeros(len(gv), dtype=bool)
    mask[top_idx] = True

    denom = float(np.mean(gv)) if len(gv) else float("nan")
    numer = float(np.mean(gv[mask])) if mask.sum() > 0 else float("nan")
    kappa = numer / denom if denom > 0 else float("nan")

    ci_lo, ci_hi = _bootstrap_kappa(gv, mask, n_boot=n_boot, seed=seed)
    rho_gate = math.sqrt(f_target / max(1e-9, f_sel * max(kappa, 1e-9)))

    return KappaResult(
        kappa=kappa,
        kappa_ci_low=ci_lo,
        kappa_ci_high=ci_hi,
        rho_gate=rho_gate,
        mean_grad_var_selected=numer,
        mean_grad_var_all=denom,
        f_sel=f_sel,
        selection_score=selection_score,
    )
