"""Position curve — Var(Q^π) vs relative position (RC-F5, plan §2B).

Bin boundaries by `relative_position ∈ [0, 1]` into deciles; report mean
Var(Q^π) per decile. Plan §2B: "If curve peaks at middle deciles, w_pos is
earned. If flat, drop w_pos from the composite score. If bimodal or
end-loaded, replace with the empirical curve as a lookup table."

Output feeds back into Stage 1's `w_pos_shape = 'lookup'` mode via a CSV
written to `experiments/sprint/position_lookup.csv` — Stage1Scorer reads
this directly in its piecewise-linear `w_pos` computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from grpocredit.common.types import OracleRecord


@dataclass
class PositionCurve:
    decile_midpoints: list[float]
    mean_var: list[float]
    std_var: list[float]
    n_per_decile: list[int]
    shape_classification: str  # 'mid_peak' | 'flat' | 'end_loaded' | 'bimodal' | 'unknown'
    peak_decile: int = -1

    def to_lookup_csv(self, path: str | Path) -> None:
        import csv

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        max_v = max(self.mean_var) if self.mean_var else 1.0
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["decile", "weight"])
            for d, v in zip(self.decile_midpoints, self.mean_var, strict=False):
                w.writerow([f"{d:.3f}", f"{(v / max_v) if max_v > 0 else 0.0:.6f}"])


def _classify_shape(
    mean_var: list[float], mid_peak_threshold: float = 0.15
) -> tuple[str, int]:
    """Classify the decile curve shape.

    - mid_peak: peak is in [3, 6] decile range AND peak > edges + threshold.
    - end_loaded: peak is in last 2 deciles.
    - bimodal: two local maxima separated by a local min.
    - flat: (max - min) / max < threshold.
    """
    if not mean_var or len(mean_var) < 3:
        return "unknown", -1
    arr = np.asarray(mean_var, dtype=float)
    lo, hi = arr.min(), arr.max()
    if hi <= 0:
        return "unknown", -1
    if (hi - lo) / hi < mid_peak_threshold:
        return "flat", int(np.argmax(arr))
    peak = int(np.argmax(arr))
    # Local maxima
    n = len(arr)
    local_max = [
        i
        for i in range(1, n - 1)
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]
    ]
    if len(local_max) >= 2:
        # check there's a dip between the two highest local maxima
        top_two = sorted(local_max, key=lambda i: arr[i], reverse=True)[:2]
        lo_i, hi_i = min(top_two), max(top_two)
        if hi_i - lo_i >= 2 and arr[(lo_i + hi_i) // 2] < 0.9 * min(arr[lo_i], arr[hi_i]):
            return "bimodal", peak
    if peak >= n - 2:
        return "end_loaded", peak
    if 3 <= peak <= 6:
        return "mid_peak", peak
    return "flat", peak


def compute_position_curve(records: list[OracleRecord], n_bins: int = 10) -> PositionCurve:
    rels = np.asarray([r.relative_position for r in records], dtype=float)
    vars_ = np.asarray(
        [r.var_q_pi if r.var_q_pi is not None else np.nan for r in records],
        dtype=float,
    )
    valid = ~np.isnan(vars_)
    rels = rels[valid]
    vars_ = vars_[valid]

    mids: list[float] = []
    means: list[float] = []
    stds: list[float] = []
    counts: list[int] = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (rels >= lo) & (rels < hi) if i < n_bins - 1 else (rels >= lo) & (rels <= hi)
        subset = vars_[mask]
        mids.append(0.5 * (lo + hi))
        means.append(float(subset.mean()) if len(subset) else 0.0)
        stds.append(float(subset.std(ddof=1)) if len(subset) > 1 else 0.0)
        counts.append(int(len(subset)))

    shape, peak = _classify_shape(means)
    return PositionCurve(
        decile_midpoints=mids,
        mean_var=means,
        std_var=stds,
        n_per_decile=counts,
        shape_classification=shape,
        peak_decile=peak,
    )
