"""Smoke tests: pydantic configs round-trip through YAML and `extends`."""

from __future__ import annotations

from pathlib import Path

from grpocredit.common.config import load_config


def test_base_config_loads() -> None:
    cfg = load_config(Path("configs/base_qwen_math.yaml"))
    assert cfg.name == "base_qwen_math"
    assert cfg.model.name_or_path.startswith("Qwen/")
    assert cfg.cascade.stage2.n_lookaheads == 4
    assert cfg.cascade.stage2.gate_mode == "binary"
    assert cfg.oracle.rollouts_per_forced_action == 32
    assert cfg.wandb.project == "grpo-voi"


def test_vineppo_extends_base() -> None:
    cfg = load_config(Path("configs/baselines/vineppo.yaml"))
    assert cfg.name == "vineppo"
    # Inherited from base
    assert cfg.model.name_or_path.startswith("Qwen/")
    # Overridden
    assert cfg.advantage.mode == "vineppo"
    assert cfg.advantage.shrinkage.mode == "none"
    assert cfg.cascade.stage1.enabled is False
    assert cfg.cascade.stage2.enabled is False
    assert cfg.oracle.rollouts_per_forced_action == 9


def test_voi_full_extends_base() -> None:
    cfg = load_config(Path("configs/proposed/voi_full.yaml"))
    assert cfg.name == "voi_full"
    assert cfg.cascade.stage2.enabled is True
    assert cfg.cascade.stage2.gate_mode == "binary"
    assert cfg.cascade.stage2.n_lookaheads == 4


def test_overrides_take_precedence() -> None:
    cfg = load_config(
        Path("configs/base_qwen_math.yaml"),
        overrides={"seed": 999, "cascade": {"stage1": {"keep_top_pct": 0.7}}},
    )
    assert cfg.seed == 999
    assert cfg.cascade.stage1.keep_top_pct == 0.7
    # Untouched fields still inherit from base
    assert cfg.cascade.stage2.n_lookaheads == 4
