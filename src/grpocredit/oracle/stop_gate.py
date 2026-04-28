"""Day-1 stop-gate classification.

This module factors the pure decision logic out of
`scripts/sprint_d1_infra_smoke.py` so it is unit-testable and — more
importantly — so the infra-vs-policy split has a single source of truth.

The classification separates two fundamentally different failure classes:

* **Infra failure** — a bug in our code/config. Blocks Day 2
  unconditionally. Exit 1. Triggers:
    - ``boundaries_max == 0`` — the boundary detector produced zero
      boundaries on *every* trajectory → detector is broken.
    - ``verifier_accuracy < threshold`` — the verifier cannot grade its
      own ground-truth → grader is broken.
* **Policy failure** — a quality signal about the starting policy, *not*
  a bug. Blocks Day 2 by default (exit 6) but is overrideable with
  ``proceed_on_policy_gate_fail=True`` for intentionally-weak debug
  policies. Triggers:
    - ``boundaries_mean < threshold`` — policy produces short CoTs
      (expected for concise-solver models like rho-1b-sft-GSM8K, which
      averages ~2 boundaries/traj on GSM8K because problems are 2-3
      steps; not a bug, a model-trained-distribution property).
    - ``fraction_informative < threshold`` — §5 group-variance gate
      (sft_warmup_plan.md §5).

``boundaries_mean`` used to be classified as infra, which created false
positives on small/short-CoT models. The detector-health check
(``boundaries_max``) is the honest infra signal — if *any* trajectory
segmented, the detector works; the mean is a policy property.

See SERVER2_RUNBOOK.md §2.5 for the operational context and overrides.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class StopGateDecision:
    """Verdict returned by :func:`classify_stop_gate`.

    Attributes:
        infra_fail: True iff the boundary detector produced nothing
            anywhere (``boundaries_max == 0``) OR the verifier failed to
            grade its own ground truth (``verifier_accuracy < min``).
            Both are code/config bugs and always block Day 2.
        policy_fail: True iff a policy-distribution-dependent gate
            failed: short CoTs (``boundaries_mean < min``) OR the §5
            group-variance gate (``fraction_informative < threshold``).
            These are quality signals about π_ref, not bugs.
        proceed_on_policy_gate_fail: The caller's override flag. Echoed
            back so the serialised report records whether the gate was
            waived (important for audit trails in GATE_REPORT.md).
        effective_stop: True iff Day 2 should NOT run. Equals
            ``infra_fail or (policy_fail and not proceed_on_policy_gate_fail)``.
        exit_code: 0 on pass (or waived policy-fail), 1 on infra_fail,
            6 on policy_fail (when not waived). Matches the exit-code
            table in SERVER2_RUNBOOK.md §2.
        reasons: Human-readable list of all triggered sub-checks, e.g.
            ``["boundaries_mean=2.3 < 3.0 (policy)", ...]``. Used for
            logging and the printed summary.
    """

    infra_fail: bool
    policy_fail: bool
    proceed_on_policy_gate_fail: bool
    effective_stop: bool
    exit_code: int
    reasons: tuple[str, ...]


def classify_stop_gate(
    *,
    boundaries_mean: float,
    boundaries_max: int,
    verifier_accuracy: float,
    gv_pass: Optional[bool],
    min_boundaries: float,
    min_verifier_accuracy: float,
    proceed_on_policy_gate_fail: bool,
) -> StopGateDecision:
    """Classify a Day-1 run's gate outcome into infra vs policy failure.

    Args:
        boundaries_mean: mean boundary count per trajectory (smoke-test
            trajectories, not the gate probe rollouts). Policy-class
            check: low mean means short CoTs, not a detector bug.
        boundaries_max: max boundary count across trajectories. Infra-
            class check: ``0`` means the detector never segmented
            anything, which can only be a detector bug.
        verifier_accuracy: verifier accuracy on the ground-truth probe.
        gv_pass: whether the §5 group-variance gate passed. ``None``
            means the gate wasn't run (e.g. ``group_variance_probe_size
            <= 0``); that case counts as "no policy info, not a fail".
        min_boundaries: below this, ``boundaries_mean`` is a **policy**
            failure (default in the script: 3.0). Calibrated for 7B
            MATH-style CoTs; per-policy overrides live in YAML configs
            or the ``--stop-gate-min-boundaries`` CLI flag.
        min_verifier_accuracy: below this, the verifier is broken
            (default in the script: 0.9). Infra-class.
        proceed_on_policy_gate_fail: if True, a *policy-only* failure
            does not block; ``exit_code`` collapses from 6 to 0. Infra
            failures still block.

    Returns:
        :class:`StopGateDecision` with the infra/policy split, the
        waiver flag, the exit code to propagate, and a tuple of
        human-readable reason strings for logging.
    """
    reasons: list[str] = []

    # ── Infra checks ─────────────────────────────────────────────
    detector_broken = boundaries_max == 0
    if detector_broken:
        reasons.append(
            f"boundaries_max={boundaries_max} — detector produced zero "
            f"boundaries on every trajectory (INFRA)"
        )
    verifier_broken = verifier_accuracy < min_verifier_accuracy
    if verifier_broken:
        reasons.append(
            f"verifier_accuracy={verifier_accuracy:.3f} < "
            f"{min_verifier_accuracy} — grader cannot score its own "
            f"ground truth (INFRA)"
        )
    infra_fail = detector_broken or verifier_broken

    # ── Policy checks ────────────────────────────────────────────
    short_cots = boundaries_mean < min_boundaries
    if short_cots:
        reasons.append(
            f"boundaries_mean={boundaries_mean:.2f} < {min_boundaries} "
            f"— policy produces short CoTs, oracle stats will be noisier "
            f"(POLICY)"
        )
    gv_fail = (gv_pass is not None) and (not gv_pass)
    if gv_fail:
        reasons.append(
            "group-variance gate: fraction_informative below threshold "
            "(POLICY)"
        )
    policy_fail = short_cots or gv_fail

    effective_stop = infra_fail or (policy_fail and not proceed_on_policy_gate_fail)

    if infra_fail:
        exit_code = 1
    elif policy_fail and not proceed_on_policy_gate_fail:
        exit_code = 6
    else:
        exit_code = 0

    return StopGateDecision(
        infra_fail=infra_fail,
        policy_fail=policy_fail,
        proceed_on_policy_gate_fail=proceed_on_policy_gate_fail,
        effective_stop=effective_stop,
        exit_code=exit_code,
        reasons=tuple(reasons),
    )
