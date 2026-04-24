"""Wandb wrapper used by every executable script.

Design points:
- `init_wandb` always returns a `WandbRun` — if wandb isn't installed or is
  disabled, the returned run is a no-op stub. Callers never have to guard.
- Config is logged exhaustively on init (pydantic `model_dump`).
- Artifacts are first-class: `run.log_artifact(path, artifact_type)` uploads
  files produced by sprint / oracle scripts.
- Offline mode is respected via env `WANDB_MODE=offline` *or* the config's
  `wandb.mode = 'offline'`; we don't re-invent the switch.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from grpocredit.common.config import ExperimentConfig, WandbConfig

log = logging.getLogger(__name__)

try:
    import wandb as _wandb

    _HAS_WANDB = True
except Exception:  # pragma: no cover
    _wandb = None  # type: ignore[assignment]
    _HAS_WANDB = False


@dataclass
class WandbRun:
    """Thin wrapper that degrades to a no-op if wandb is disabled / missing."""

    handle: Any = None  # wandb.sdk.wandb_run.Run | None
    enabled: bool = False
    run_dir: Path | None = None

    @property
    def url(self) -> str:
        return getattr(self.handle, "url", "") or ""

    @property
    def id(self) -> str:
        return getattr(self.handle, "id", "") or ""

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self.enabled and self.handle is not None:
            if step is not None:
                self.handle.log(metrics, step=step)
            else:
                self.handle.log(metrics)

    def log_summary(self, **kwargs: Any) -> None:
        if self.enabled and self.handle is not None:
            for k, v in kwargs.items():
                self.handle.summary[k] = v

    def log_artifact(
        self,
        path: str | Path,
        artifact_type: str = "dataset",
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled or self.handle is None or _wandb is None:
            return
        path = Path(path)
        if not path.exists():
            log.warning("log_artifact: %s does not exist, skipping", path)
            return
        art_name = name or f"{artifact_type}-{path.stem}"
        art = _wandb.Artifact(art_name, type=artifact_type, metadata=metadata or {})
        if path.is_dir():
            art.add_dir(str(path))
        else:
            art.add_file(str(path))
        self.handle.log_artifact(art)

    def log_table(self, key: str, columns: list[str], rows: list[list[Any]]) -> None:
        if not self.enabled or self.handle is None or _wandb is None:
            return
        table = _wandb.Table(columns=columns, data=rows)
        self.handle.log({key: table})

    def finish(self, exit_code: int = 0) -> None:
        if self.enabled and self.handle is not None:
            self.handle.finish(exit_code=exit_code)


def init_wandb(
    config: ExperimentConfig,
    *,
    run_name: str | None = None,
    wandb_cfg: WandbConfig | None = None,
    extra_config: dict[str, Any] | None = None,
) -> WandbRun:
    """Initialize a wandb run; return a `WandbRun` wrapper that never raises."""
    wcfg = wandb_cfg or config.wandb
    env_mode = os.environ.get("WANDB_MODE")
    mode = env_mode or wcfg.mode

    if mode == "disabled" or not _HAS_WANDB:
        if not _HAS_WANDB:
            log.warning("wandb package not available; running in no-op mode")
        return WandbRun(enabled=False)

    assert _wandb is not None

    payload = config.model_dump(mode="json")
    if extra_config:
        payload = {**payload, **extra_config}

    name = run_name or config.name
    tags = list(wcfg.tags)
    handle = _wandb.init(
        project=wcfg.project,
        entity=wcfg.entity,
        name=name,
        tags=tags,
        group=wcfg.group,
        job_type=wcfg.job_type,
        mode=mode,
        config=payload,
        dir=str(Path(config.output_dir) / ".wandb"),
        reinit=True,
    )
    run_dir = Path(handle.dir) if handle is not None else None
    return WandbRun(handle=handle, enabled=True, run_dir=run_dir)
