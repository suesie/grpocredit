"""Shared dataclasses used across rollout / cascade / advantage / oracle layers.

Every offline experiment ultimately produces a stream of `Trajectory` → `Boundary`
records, plus (for the Q-variance oracle) `OracleRecord`s that bolt forced-action
Q-estimates onto specific boundaries. Keeping these lean and pure-Python makes
serialisation to JSONL trivial.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RolloutResult:
    """One sampled continuation from a prompt or from a mid-trajectory state.

    `logprob_sum` is on the *generated* tokens only, matching GRPO's ratio term.
    `token_entropies` has length `len(token_ids)` when the backend exposes
    per-step next-token entropies; otherwise empty.
    """

    prompt: str
    prompt_token_ids: list[int]
    response_text: str
    token_ids: list[int]
    logprobs: list[float]
    logprob_sum: float
    token_entropies: list[float] = field(default_factory=list)
    finish_reason: str = ""
    backend: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def num_generated_tokens(self) -> int:
        return len(self.token_ids)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Trajectory:
    """A single prompt + completion scored against a verifiable answer."""

    trajectory_id: str
    prompt_id: str
    prompt: str
    prompt_token_ids: list[int]
    response_text: str
    token_ids: list[int]
    logprobs: list[float]
    token_entropies: list[float]
    reward: float
    correct: bool
    ground_truth_answer: str
    extracted_answer: str = ""
    finish_reason: str = ""
    group_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.token_ids)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Boundary:
    """One candidate boundary inside a `Trajectory`.

    `token_position` = index into `trajectory.token_ids` of the *first* token
    of the segment **after** the boundary. Equivalently: number of tokens in
    the prefix up to and including the boundary marker. A boundary at position
    `k` splits the trajectory into `token_ids[:k]` (prefix, i.e. `s_b`) and
    `token_ids[k:]` (continuation).

    `char_position` is the analogous index into `response_text` (UTF-8).
    """

    trajectory_id: str
    boundary_idx: int
    token_position: int
    char_position: int
    kind: str  # paragraph / step / therefore / however / newline / punct / byte
    marker_text: str = ""
    # Cascade-populated fields (None if a given stage hasn't run)
    h_token: float | None = None
    w_pos: float | None = None
    s1: float | None = None
    h_sem: float | None = None
    s2: float | None = None
    cusum_abs: float | None = None
    selected: bool = False
    stage_stopped_at: int = 0  # 0 = passed all stages, 1 = cut by Stage 1, ...
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ForcedActionResult:
    """Result of running K terminal rollouts with a forced first-action."""

    boundary_id: str
    first_token_id: int
    first_token_str: str
    pi_first_token: float  # π(a | s_b) from the policy at the boundary
    rewards: list[float]
    response_texts: list[str]
    num_correct: int = 0

    @property
    def q_hat(self) -> float:
        if not self.rewards:
            return float("nan")
        return sum(self.rewards) / len(self.rewards)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OracleRecord:
    """One boundary's full oracle bundle.

    `forced_action_results` is the list over top-M first-actions. `tail_result`
    is the optional sampled-tail stratum (None if coverage c ≥ 0.9).
    """

    boundary_id: str
    trajectory_id: str
    boundary_idx: int
    token_position: int
    relative_position: float  # token_position / trajectory_length, in [0, 1]
    trajectory_length: int
    coverage_c: float
    forced_action_results: list[ForcedActionResult]
    tail_result: ForcedActionResult | None = None
    # Cheap signals already computed
    h_token: float | None = None
    w_pos: float | None = None
    s1: float | None = None
    h_sem: float | None = None
    s2: float | None = None
    # Oracle-derived
    v_hat: float | None = None
    var_q_pi: float | None = None  # head-truncated unless tail_result is set
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        return out
