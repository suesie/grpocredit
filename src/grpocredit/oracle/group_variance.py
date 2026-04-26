"""Group-variance gate at step 0.

Per `research_plan/sft_warmup_plan.md` §5 (lines 246, 250-255), the central
go/no-go check before any RL run is the **group-variance fraction at step 0**:

    Sample N prompts. For each, generate G rollouts. Score each rollout.
    Count the fraction of groups where Var(reward) > 0 (i.e., not all
    rollouts got the same reward).

    PASS:  ≥ 0.5     informative-group fraction
    FAIL:  < 0.5     start is too saturated (e.g., RL on top of GRPO'd
                     model with pass@1 ≈ 0.95) or too weak (no group has
                     any positive reward).

The same metric is what motivates the VoI framing for the paper itself: VoI
is most valuable precisely when informative groups are scarce. So the gate
output also serves as a paper figure ("fraction of GRPO groups with
informative advantage at step 0 across starting policies").

This module exposes the pure compute (testable, no vLLM) plus a thin wrapper
that wires it into the sprint-d1 pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass
class GroupVarianceReport:
    n_groups: int
    n_informative: int
    fraction_informative: float
    n_groups_all_correct: int        # all rewards == max
    n_groups_all_wrong: int          # all rewards == 0 (or all == min)
    mean_group_reward_mean: float    # average across groups of group-mean reward
    mean_group_reward_std: float     # average across groups of group-std reward

    def to_dict(self) -> dict:
        return {
            "n_groups": self.n_groups,
            "n_informative": self.n_informative,
            "fraction_informative": self.fraction_informative,
            "n_groups_all_correct": self.n_groups_all_correct,
            "n_groups_all_wrong": self.n_groups_all_wrong,
            "mean_group_reward_mean": self.mean_group_reward_mean,
            "mean_group_reward_std": self.mean_group_reward_std,
        }


def compute_group_variance_report(
    grouped_rewards: Sequence[Sequence[float]],
    informative_eps: float = 1e-12,
    correct_threshold: float = 1.0,
    wrong_threshold: float = 0.0,
) -> GroupVarianceReport:
    """Pure-function group-variance compute.

    Args:
        grouped_rewards: list of groups, each a list of per-rollout rewards.
            Group sizes do not need to be equal (degenerate groups with size
            < 2 contribute zero variance and zero learning signal — they are
            counted as non-informative).
        informative_eps: a group counts as informative iff its empirical
            std exceeds this (so floating-point ties don't get classified
            as informative). Default 1e-12.
        correct_threshold / wrong_threshold: the verifier produces 0/1 for
            our pipeline, but the function accepts arbitrary scalar rewards;
            an "all-correct" group is one where every reward equals
            `correct_threshold`, "all-wrong" likewise for `wrong_threshold`.

    Returns:
        GroupVarianceReport with both the headline `fraction_informative`
        and breakdown stats useful for the paper figure.
    """
    if not grouped_rewards:
        return GroupVarianceReport(0, 0, 0.0, 0, 0, 0.0, 0.0)

    import numpy as np

    n_groups = len(grouped_rewards)
    n_informative = 0
    n_all_correct = 0
    n_all_wrong = 0
    group_means: List[float] = []
    group_stds: List[float] = []

    for g in grouped_rewards:
        if len(g) == 0:
            continue
        arr = np.asarray(g, dtype=np.float64)
        std = float(arr.std())
        mean = float(arr.mean())
        group_means.append(mean)
        group_stds.append(std)
        if std > informative_eps:
            n_informative += 1
        else:
            # Saturated group — figure out which way.
            v0 = float(arr[0])
            if abs(v0 - correct_threshold) < 1e-9:
                n_all_correct += 1
            elif abs(v0 - wrong_threshold) < 1e-9:
                n_all_wrong += 1
            # otherwise: degenerate but on a non-binary value (rare for our
            # binary verifier) — counted as non-informative but neither bin.

    fraction_informative = n_informative / n_groups if n_groups else 0.0
    mean_group_reward_mean = float(sum(group_means) / len(group_means)) if group_means else 0.0
    mean_group_reward_std = float(sum(group_stds) / len(group_stds)) if group_stds else 0.0

    return GroupVarianceReport(
        n_groups=n_groups,
        n_informative=n_informative,
        fraction_informative=fraction_informative,
        n_groups_all_correct=n_all_correct,
        n_groups_all_wrong=n_all_wrong,
        mean_group_reward_mean=mean_group_reward_mean,
        mean_group_reward_std=mean_group_reward_std,
    )


def grouped_rewards_from_runner_output(
    grouped_rollouts: Iterable[Iterable],
    verifier,
    ground_truth_answers: Sequence[str],
) -> List[List[float]]:
    """Score `runner.generate_from_prompts(...)` output through `MathVerifier`
    and return per-group reward lists. One list per prompt; each inner list
    has G floats (one per rollout). Used by sprint_d1.

    Args:
        grouped_rollouts: outer iterable yields per-prompt iterables of
            `RolloutResult`-like objects with a `.response_text` attribute.
        verifier: a `MathVerifier`.
        ground_truth_answers: aligned with the outer iterable, one
            ground-truth string per prompt.
    """
    out: List[List[float]] = []
    for group, gt in zip(grouped_rollouts, ground_truth_answers, strict=False):
        rewards: List[float] = []
        for rr in group:
            v = verifier.score(rr.response_text, gt)
            rewards.append(1.0 if v.correct else 0.0)
        out.append(rewards)
    return out
