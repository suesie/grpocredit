from __future__ import annotations

import json
from pathlib import Path

from grpocredit.common.utils import (
    chunked,
    fisher_z_ci,
    read_jsonl,
    write_jsonl,
)


def test_chunked_splits_evenly() -> None:
    result = list(chunked(list(range(10)), 3))
    assert result == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]


def test_fisher_z_ci_contains_rho() -> None:
    lo, hi = fisher_z_ci(0.3, n=500)
    assert lo < 0.3 < hi
    # 95% CI half-width ~ 1.96 / sqrt(497) ≈ 0.088 in z-space → ~0.08 in r-space
    assert hi - lo < 0.20


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    records = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    p = tmp_path / "a.jsonl"
    n = write_jsonl(p, records)
    assert n == 2
    read_back = read_jsonl(p)
    assert read_back == records
    # Raw file is one JSON per line
    lines = p.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    for ln in lines:
        json.loads(ln)
