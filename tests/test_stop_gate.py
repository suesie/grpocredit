"""Unit tests for the Day-1 stop-gate infra-vs-policy classification.

Covers the three-way decision that `scripts/sprint_d1_infra_smoke.py` makes
at the end of Day 1:

* PASS: all checks clear.
* INFRA fail: ``boundaries_max == 0`` (detector produced nothing on any
  trajectory → detector is broken) OR ``verifier_accuracy < min`` (grader
  cannot score its own ground truth → grader is broken). Always blocks
  Day 2 (exit 1). **Not waivable** by ``--proceed-on-policy-gate-fail``.
* POLICY fail: ``boundaries_mean < min`` (policy produces short CoTs — a
  distribution property, e.g. rho-1b on GSM8K averages ~2) OR the §5
  group-variance gate failed. Blocks Day 2 by default (exit 6), but
  collapsible to exit 0 via ``proceed_on_policy_gate_fail=True``.

Historically ``boundaries_mean`` was classified as infra, which produced
false positives on short-CoT models like ``rho-1b-sft-GSM8K`` whose
GSM8K trajectories average ~2.3 boundaries (the threshold is 3.0). The
detector-health check (``boundaries_max > 0``) is the honest infra
signal: if *any* trajectory segmented, the detector works; the mean is
a policy-trained-distribution property.

These tests pin the contract so the override and exit-code semantics
cannot drift silently; they matter because `run_oracle.sh` relies on
exit codes 1/6/0 to decide whether to continue to Day 2.
"""

from __future__ import annotations

from grpocredit.oracle.stop_gate import classify_stop_gate


# Convenience: the defaults used in sprint_d1_infra_smoke.py — if you
# change these in one place without the other, the tests below should
# still be conceptually valid.
MIN_B = 3.0
MIN_V = 0.9


def _c(**kw):
    """classify_stop_gate with sensible defaults; override via kwargs."""
    defaults = dict(
        boundaries_mean=6.0,
        boundaries_max=14,
        verifier_accuracy=0.99,
        gv_pass=True,
        min_boundaries=MIN_B,
        min_verifier_accuracy=MIN_V,
        proceed_on_policy_gate_fail=False,
    )
    defaults.update(kw)
    return classify_stop_gate(**defaults)


# ── Happy path ─────────────────────────────────────────────────────────


def test_all_pass_exits_zero():
    d = _c()
    assert not d.infra_fail
    assert not d.policy_fail
    assert not d.effective_stop
    assert d.exit_code == 0
    assert d.reasons == ()


def test_gv_not_run_is_not_a_failure():
    # If the gate was skipped (probe size 0), gv_pass is None and the
    # classification should NOT mark it as a policy failure.
    d = _c(gv_pass=None)
    assert not d.policy_fail
    assert not d.effective_stop
    assert d.exit_code == 0


def test_threshold_is_strict_less_than_not_leq():
    # boundaries_mean exactly at threshold is NOT a failure.
    d = _c(boundaries_mean=MIN_B, verifier_accuracy=MIN_V)
    assert not d.infra_fail
    assert not d.policy_fail
    assert d.exit_code == 0


# ── Infra: detector broken (boundaries_max == 0) ───────────────────────


def test_detector_broken_is_infra_fail_exit_1():
    # Only boundaries_max == 0 means "detector broken"; low mean alone
    # does NOT.
    d = _c(boundaries_mean=0.0, boundaries_max=0)
    assert d.infra_fail
    assert d.effective_stop
    assert d.exit_code == 1
    assert any("boundaries_max=0" in r for r in d.reasons)


def test_detector_max_one_is_not_infra():
    # Detector produced exactly one boundary across the whole run — the
    # detector is trivially working; not infra.
    d = _c(boundaries_mean=0.01, boundaries_max=1)
    assert not d.infra_fail
    # It IS a policy fail (short CoTs), though.
    assert d.policy_fail


# ── Infra: verifier broken ─────────────────────────────────────────────


def test_low_verifier_acc_is_infra_fail_exit_1():
    d = _c(verifier_accuracy=0.5)
    assert d.infra_fail
    assert d.effective_stop
    assert d.exit_code == 1
    assert any("verifier_accuracy" in r for r in d.reasons)


def test_infra_fail_not_waivable_by_override():
    # The override only affects *policy* failures. Infra failures still
    # block even when the operator asked to proceed.
    d = _c(boundaries_max=0, gv_pass=False, proceed_on_policy_gate_fail=True)
    assert d.infra_fail
    assert d.policy_fail
    assert d.effective_stop
    assert d.exit_code == 1  # infra, NOT 6 or 0

    d2 = _c(verifier_accuracy=0.5, gv_pass=False, proceed_on_policy_gate_fail=True)
    assert d2.infra_fail
    assert d2.exit_code == 1


# ── Policy: short CoTs (was incorrectly infra before) ──────────────────


def test_short_cots_is_policy_not_infra():
    # This is the exact rho-1b case: boundaries_mean=2.26, max=14.
    # Previously this was classified as infra and blocked unconditionally.
    # It should now be policy and waivable.
    d = _c(boundaries_mean=2.26, boundaries_max=14)
    assert not d.infra_fail, "short CoTs are NOT a code bug"
    assert d.policy_fail
    assert d.effective_stop  # default still blocks
    assert d.exit_code == 6  # policy, not infra
    assert any("boundaries_mean" in r for r in d.reasons)


def test_short_cots_waivable_by_override():
    # The whole point of the reclassification: rho-1b can proceed.
    d = _c(
        boundaries_mean=2.26,
        boundaries_max=14,
        proceed_on_policy_gate_fail=True,
    )
    assert not d.infra_fail
    assert d.policy_fail
    assert not d.effective_stop
    assert d.exit_code == 0


# ── Policy: group-variance gate ────────────────────────────────────────


def test_gv_fail_default_exits_6():
    d = _c(gv_pass=False)
    assert not d.infra_fail
    assert d.policy_fail
    assert d.effective_stop
    assert d.exit_code == 6


def test_gv_fail_override_collapses_to_exit_0():
    d = _c(gv_pass=False, proceed_on_policy_gate_fail=True)
    assert not d.infra_fail
    assert d.policy_fail
    assert not d.effective_stop
    assert d.exit_code == 0
    assert d.proceed_on_policy_gate_fail is True


def test_both_policy_flags_trip_together_for_rho1b():
    # The actual failure mode observed on rho-1b GSM8K test split:
    # boundaries_mean=2.26 < 3, fraction_informative=0.492 < 0.5, but
    # infra (verifier, detector-max) is clean. Both are policy-class.
    d = _c(
        boundaries_mean=2.26,
        boundaries_max=14,
        verifier_accuracy=1.0,
        gv_pass=False,
    )
    assert not d.infra_fail
    assert d.policy_fail
    assert d.exit_code == 6
    # Both reasons should be reported.
    reasons_joined = " | ".join(d.reasons)
    assert "boundaries_mean" in reasons_joined
    assert "group-variance" in reasons_joined

    # Same inputs with override → proceed to Day 2.
    d_override = _c(
        boundaries_mean=2.26,
        boundaries_max=14,
        verifier_accuracy=1.0,
        gv_pass=False,
        proceed_on_policy_gate_fail=True,
    )
    assert d_override.exit_code == 0
    assert not d_override.effective_stop


# ── Exit-code → runbook table alignment ────────────────────────────────


def test_exit_codes_match_runbook_table():
    """SERVER2_RUNBOOK.md §2 explicitly lists exit codes 1, 6, 7. This
    test pins that our classifier never produces any other code — a drift
    would quietly break run_oracle.sh's branching."""
    allowed = {0, 1, 6}
    for boundaries_mean in [0.0, 2.26, 3.0, 10.0]:
        for boundaries_max in [0, 1, 14]:
            for verifier in [0.5, 0.9, 1.0]:
                for gv_pass in [None, True, False]:
                    for override in [False, True]:
                        d = _c(
                            boundaries_mean=boundaries_mean,
                            boundaries_max=boundaries_max,
                            verifier_accuracy=verifier,
                            gv_pass=gv_pass,
                            proceed_on_policy_gate_fail=override,
                        )
                        assert d.exit_code in allowed, (
                            f"unexpected exit code {d.exit_code} for "
                            f"mean={boundaries_mean}, max={boundaries_max}, "
                            f"verifier={verifier}, gv_pass={gv_pass}, "
                            f"override={override}"
                        )
