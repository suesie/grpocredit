"""Pydantic-validated YAML configs.

One `ExperimentConfig` is built by `load_config(path, overrides=...)`. Every
nested block is its own pydantic model — makes it easy to pass just the
`RolloutConfig` into the vLLM runner without dragging the whole tree.

Config files support a top-level `extends:` key (string or list) that names
other YAML files whose contents are shallow-merged before the current file's
values are applied. This matches the `configs/baselines/` layout in the plan.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Base(BaseModel):
    """Forbid extra keys by default — catches typos in YAML early."""

    model_config = ConfigDict(extra="forbid", frozen=False)


class ModelConfig(_Base):
    name_or_path: str = "Qwen/Qwen2.5-Math-7B-Instruct"
    tokenizer_name_or_path: str | None = None
    dtype: str = "bfloat16"
    trust_remote_code: bool = True
    chat_template: str | None = None  # path or inline

    @property
    def tokenizer_id(self) -> str:
        return self.tokenizer_name_or_path or self.name_or_path


class RolloutConfig(_Base):
    backend: str = "vllm"  # 'vllm' | 'hf'
    max_new_tokens: int = 1024
    temperature: float = 0.9
    top_p: float = 0.95
    top_k: int = -1
    stop: list[str] = Field(default_factory=list)
    seed: int = 42
    logprobs: int = 5  # top-k logprobs returned per step; used for entropy estimates
    n_per_prompt: int = 8  # group size G
    batch_size: int = 64
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    swap_space_gb: int = 4
    enable_prefix_caching: bool = True
    # Lookahead rollouts for Stage 2
    lookahead_max_new_tokens: int = 30
    lookahead_temperature: float = 1.0
    # When True AND `n > 1`, the runner fans out into N single-sample
    # requests with seeds `[base, base+1, …]` (VinePPO convention —
    # bit-reproducible, slower). When False (default), the runner drops
    # the per-request seed on `n > 1` calls (verl/TRL/OpenRLHF convention
    # — faster, not bit-reproducible). See the `vllm_runner` module
    # docstring and runbook §2.3 for the full rationale.
    deterministic_n: bool = False


class BoundaryConfig(_Base):
    """Syntactic delimiter regexes inherit from VinePPO §3.2.2."""

    paragraph_break: str = r"\n\s*\n"
    step_marker: str = r"(?m)^(?:Step\s*\d+[:.)]|\d+[.)]\s)"
    logical_markers: list[str] = Field(
        default_factory=lambda: [
            r"\bTherefore[,.]?\b",
            r"\bHowever[,.]?\b",
            r"\bBut[,.]?\b",
            r"\bHence[,.]?\b",
            r"\bThus[,.]?\b",
            r"\bSo[,.]?\b",
            r"\\boxed\s*\{",  # MATH-style answer marker
        ]
    )
    # Drift fallback (§3.2.2)
    sentence_punct: str = r"(?<=[.!?])\s+(?=[A-Z])"
    byte_budget_stride: int = 128  # every-N-tokens fallback
    min_boundaries_for_drift_fallback: int = 3
    max_boundaries_per_trajectory: int = 30
    min_tokens_between_boundaries: int = 8


class Stage0Config(_Base):
    enabled: bool = True
    # Dropped groups: 0/G and G/G success-rate; kept: intermediate
    # (Matches DAPO dynamic sampling.)


class Stage1Config(_Base):
    enabled: bool = True
    # w_pos shape: 'tent' (1 - |2t/T - 1|), 'gaussian' (centred at T/2),
    # 'lookup' (path to CSV from position_curve.py), 'uniform' (1 everywhere)
    w_pos_shape: str = "tent"
    w_pos_gaussian_sigma: float = 0.25  # fraction of T
    w_pos_lookup_path: str | None = None
    keep_top_pct: float = 0.5
    use_collision_entropy: bool = False  # True → 1 - ||π||^2 instead of Shannon H


class Stage2Config(_Base):
    enabled: bool = True
    n_lookaheads: int = 4  # K_LA
    lookahead_max_new_tokens: int = 30
    lookahead_temperature: float = 1.0
    encoder: str = "sentence-transformers/sentence-t5-base"
    encoder_batch_size: int = 32
    cosine_threshold: float = 0.85
    gate_mode: str = "binary"  # 'binary' (D2 default at K_LA=4) | 'continuous'
    keep_top_pct: float = 0.3  # of Stage-1 survivors
    epsilon_random: float = 0.05  # coverage hedge, added to selected set
    nli_fallback_model: str | None = "microsoft/deberta-v2-xxlarge-mnli"


class CusumConfig(_Base):
    enabled: bool = False
    reference_model_path: str | None = None  # π_ref for implicit-reward term
    window_size: int = 15
    keep_top_pct: float = 0.2


class CascadeConfig(_Base):
    stage0: Stage0Config = Field(default_factory=Stage0Config)
    stage1: Stage1Config = Field(default_factory=Stage1Config)
    stage2: Stage2Config = Field(default_factory=Stage2Config)
    cusum: CusumConfig = Field(default_factory=CusumConfig)


class ShrinkageConfig(_Base):
    mode: str = "james_stein"  # 'james_stein' | 'se' | 'none'
    james_stein_tau: float = 4.0  # α(M) = M / (M + tau)


class AdvantageConfig(_Base):
    mode: str = "voi"  # 'grpo' | 'vineppo' | 'voi' | 'random_cascade'
    token_assignment: str = "uniform"  # only 'uniform' supported in §3.3.3 default
    shrinkage: ShrinkageConfig = Field(default_factory=ShrinkageConfig)
    length_aggregation: str = "per_sequence"  # 'per_sequence' | 'token' | 'dr_grpo' | 'constant'


class OracleConfig(_Base):
    n_trajectories: int = 100
    boundaries_per_trajectory: int = 5
    top_m_actions: int = 6
    rollouts_per_forced_action: int = 32  # K
    include_tail_stratum: bool = True
    tail_stratum_size: int = 16
    coverage_threshold_for_tail: float = 0.9  # run tail if c < this
    # Forward-entropy window: H_fwd(b) = mean(H(π(·|s_{b+k})) for k=0..h_fwd_k-1)
    h_fwd_k: int = 10
    # Concordance-specific
    concordance_lookaheads: int = 4  # K_LA
    concordance_terminal_temperature: float = 0.9
    concordance_min_boundaries: int = 500


class DataConfig(_Base):
    train_datasets: list[str] = Field(default_factory=lambda: ["gsm8k", "math"])
    eval_datasets: list[str] = Field(default_factory=lambda: ["math", "gsm8k"])
    ood_datasets: list[str] = Field(default_factory=lambda: ["aime24", "olympiadbench", "math500"])
    split: str = "train"  # used by sprint scripts
    n_prompts: int | None = None  # None → full split
    prompt_template: str = "math_instruct"  # key into grpocredit.rollout.datasets.TEMPLATES


class WandbConfig(_Base):
    project: str = "grpo-voi"
    entity: str | None = None
    mode: str = "online"  # 'online' | 'offline' | 'disabled'
    tags: list[str] = Field(default_factory=list)
    group: str | None = None
    job_type: str | None = None


class ExperimentConfig(_Base):
    name: str = "unnamed"
    seed: int = 42
    output_dir: str = "experiments/sprint"
    device: str = "cuda"

    model: ModelConfig = Field(default_factory=ModelConfig)
    rollout: RolloutConfig = Field(default_factory=RolloutConfig)
    boundary: BoundaryConfig = Field(default_factory=BoundaryConfig)
    cascade: CascadeConfig = Field(default_factory=CascadeConfig)
    advantage: AdvantageConfig = Field(default_factory=AdvantageConfig)
    oracle: OracleConfig = Field(default_factory=OracleConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)

    @model_validator(mode="after")
    def _sync_tokenizer(self) -> ExperimentConfig:
        if self.model.tokenizer_name_or_path is None:
            self.model.tokenizer_name_or_path = self.model.name_or_path
        return self


# ─── aliases for tidier imports in layer code ─────────────────────────────
BaseConfig = ExperimentConfig


# ─── config loading ───────────────────────────────────────────────────────
def _shallow_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base`. Dicts merge, everything else replaces."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _shallow_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level of {path} must be a mapping, got {type(data).__name__}")
    return data


def _resolve_extends(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    """Apply `extends:` chain, deepest-first, returning merged dict."""
    seen = seen or set()
    path = path.resolve()
    if path in seen:
        raise ValueError(f"Circular extends in {path}")
    seen = seen | {path}

    data = _load_yaml(path)
    extends = data.pop("extends", None)
    if extends is None:
        return data

    extends_list = [extends] if isinstance(extends, str) else list(extends)
    merged: dict[str, Any] = {}
    for ext in extends_list:
        ext_path = (path.parent / ext).resolve()
        merged = _shallow_merge(merged, _resolve_extends(ext_path, seen))
    return _shallow_merge(merged, data)


def load_config(
    path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> ExperimentConfig:
    """Load a YAML config, resolve `extends:`, apply overrides, validate."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    data = _resolve_extends(path)
    if overrides:
        data = _shallow_merge(data, overrides)
    return ExperimentConfig.model_validate(data)
