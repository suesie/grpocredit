"""Rollout-diversity sentinel.

Before we trust any `n > 1` sampling result (GRPO groups, forced-action
rollouts for Q^π-variance, Stage-2 lookaheads) we want a fast, unambiguous
check that the runner is *actually* producing diverse samples and not
silently collapsing to duplicates.

The latter failure mode is the one that broke the Day-1 group-variance gate
on `rho-1b-sft-GSM8K`: `mean_group_reward_std=0.0` across all 256 probe
groups. That metric can go to zero for two reasons — either every prompt is
fully saturated (pass@1 ∈ {0,1} identically per prompt) or every group's
rollouts are byte-identical. This sentinel disambiguates by looking directly
at the raw response *texts*, not the rewards.

Compute is a pure function so it can be unit-tested without vLLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass
class RolloutDiversityReport:
    n_groups: int
    group_size: int
    n_groups_all_identical: int
    n_groups_fully_unique: int
    mean_unique_fraction: float  # mean over groups of (n_unique / group_size)

    def to_dict(self) -> dict:
        return {
            "n_groups": self.n_groups,
            "group_size": self.group_size,
            "n_groups_all_identical": self.n_groups_all_identical,
            "n_groups_fully_unique": self.n_groups_fully_unique,
            "mean_unique_fraction": self.mean_unique_fraction,
        }


def compute_diversity_report(grouped_texts: Sequence[Sequence[str]]) -> RolloutDiversityReport:
    """Count per-group distinct response texts.

    A healthy rollout backend at temperature 0.9 with `n > 1` will produce
    almost all fully-unique groups even on a strong base model — two sampled
    trajectories over ~100 tokens collide with negligible probability unless
    the RNG path is pinned (in which case every group is all-identical, the
    dead giveaway).

    The threshold the Day-1 sentinel uses is intentionally lax: we only flag
    a failure when the *dominant* outcome is "group all-identical". That way
    we don't spuriously alarm on a saturated policy whose distribution
    genuinely concentrates.
    """
    if not grouped_texts:
        return RolloutDiversityReport(0, 0, 0, 0, 0.0)

    n_groups = len(grouped_texts)
    group_size = max((len(g) for g in grouped_texts), default=0)
    n_all_identical = 0
    n_fully_unique = 0
    unique_fracs: list[float] = []

    for g in grouped_texts:
        if len(g) <= 1:
            # Degenerate; count as neither signal direction.
            unique_fracs.append(1.0 if len(g) == 1 else 0.0)
            continue
        uniq = len(set(g))
        if uniq == 1:
            n_all_identical += 1
        if uniq == len(g):
            n_fully_unique += 1
        unique_fracs.append(uniq / len(g))

    mean_unique = sum(unique_fracs) / len(unique_fracs) if unique_fracs else 0.0
    return RolloutDiversityReport(
        n_groups=n_groups,
        group_size=group_size,
        n_groups_all_identical=n_all_identical,
        n_groups_fully_unique=n_fully_unique,
        mean_unique_fraction=mean_unique,
    )


def assert_diverse_rollouts(
    grouped_texts: Sequence[Sequence[str]],
    *,
    min_mean_unique_fraction: float = 0.5,
    max_all_identical_fraction: float = 0.5,
) -> RolloutDiversityReport:
    """Raise if `grouped_texts` looks like a collapsed rollout batch.

    Args:
        grouped_texts: per-group list of response strings; outer length is
            the number of groups, inner length is the group size.
        min_mean_unique_fraction: average `n_unique / group_size` must
            exceed this. 0.5 means "on average, at least half the samples
            in each group are distinct". Healthy temp=0.9 sampling sits
            near 1.0 on this metric.
        max_all_identical_fraction: fraction of groups that are entirely
            one unique value must NOT exceed this. A healthy backend sits
            near 0.0; the `rho-1b` collapse was 1.0.

    Returns the report so the caller can log it to wandb.
    """
    report = compute_diversity_report(grouped_texts)
    fraction_collapsed = (
        report.n_groups_all_identical / report.n_groups if report.n_groups else 0.0
    )
    if (
        report.mean_unique_fraction < min_mean_unique_fraction
        or fraction_collapsed > max_all_identical_fraction
    ):
        raise RolloutDiversityError(
            f"Rollout-diversity sentinel FAILED: "
            f"mean_unique_fraction={report.mean_unique_fraction:.3f} "
            f"(need ≥ {min_mean_unique_fraction:.2f}), "
            f"fraction_all_identical={fraction_collapsed:.3f} "
            f"(need ≤ {max_all_identical_fraction:.2f}). "
            f"This is almost always the `n>1 + fixed seed + prefix_caching` "
            f"collapse described in grpocredit.rollout.vllm_runner's docstring. "
            f"Raw report: {report.to_dict()}"
        )
    return report


class RolloutDiversityError(RuntimeError):
    """Raised by the Day-1 sentinel when rollouts are not actually diverse."""


def diversity_probe(
    runner,
    *,
    probe_prompts: Iterable[str],
    n_per_prompt: int = 4,
    max_new_tokens: int = 64,
    seed: int | None = None,
) -> list[list[str]]:
    """Convenience wrapper: sample a small batch from the runner and return
    per-group response-text lists suitable for `assert_diverse_rollouts`.

    Small `n_per_prompt` and `max_new_tokens` keep this under ~2 s even on
    a 7B model. Called at the top of Day-1 to fail fast before the 256×8
    gate consumes real GPU time.
    """
    prompts = list(probe_prompts)
    groups = runner.generate_from_prompts(
        prompts,
        n_per_prompt=n_per_prompt,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )
    return [[rr.response_text for rr in group] for group in groups]
