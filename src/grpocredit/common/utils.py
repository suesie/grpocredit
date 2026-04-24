"""Shared utilities: seeding, path helpers, JSONL I/O, chunked iteration."""

from __future__ import annotations

import json
import logging
import os
import random
from collections.abc import Iterable, Iterator
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and torch (CPU + CUDA) RNGs. Safe if torch not installed."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _default_encoder(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, Path):
        return str(obj)
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
    except ImportError:
        pass
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def write_jsonl(path: str | Path, records: Iterable[Any]) -> int:
    path = Path(path)
    ensure_dir(path.parent)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, default=_default_encoder, ensure_ascii=False))
            f.write("\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_json(path: str | Path, obj: Any, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, default=_default_encoder, ensure_ascii=False, indent=indent)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def chunked(seq: list[T], n: int) -> Iterator[list[T]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def fisher_z_ci(rho: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Fisher z-transform 95% CI for a sample correlation coefficient.

    Used by §3.2.5 — report CI on every reported ρ.
    """
    import math

    from scipy import stats

    if n < 4:
        return (float("nan"), float("nan"))
    rho_clipped = max(min(rho, 0.9999), -0.9999)
    z = 0.5 * math.log((1 + rho_clipped) / (1 - rho_clipped))
    se = 1.0 / math.sqrt(n - 3)
    zcrit = stats.norm.ppf(1 - alpha / 2)
    z_lo, z_hi = z - zcrit * se, z + zcrit * se

    def _inv(zv: float) -> float:
        return (math.exp(2 * zv) - 1) / (math.exp(2 * zv) + 1)

    return (_inv(z_lo), _inv(z_hi))


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    import numpy as np

    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
