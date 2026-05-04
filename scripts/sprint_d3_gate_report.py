"""Sprint Day 3 — gate report.

Reads:
  experiments/sprint/emb_var_summary.json
  experiments/sprint/oracle_summary.json
  experiments/sprint/oracle_correlations.json
  experiments/sprint/oracle_kappa.txt
  experiments/sprint/day1_group_variance.json   (sft_warmup_plan.md §5 gate)

Writes:
  experiments/sprint/GATE_REPORT.md   — human-readable decision table
  experiments/sprint/gate_decision.json — machine-readable

Prints a decision table and exits with:
  0  — all gates pass → proceed with main plan
  2  — marginal (1 gate marginal) → proceed with caveats
  3  — 2+ marginal → pilot training needed
  4  — hard fail on concordance → pivot to Plan B (§8 of plan)
  5  — sprint outputs missing
  6  — group-variance gate failed → wrong starting policy
       (sft_warmup_plan.md §5; switch `π_ref` and re-run)

The exit code lets `scripts/main_train.sh` decide whether to launch the main
experiment, fork into Plan B, or refuse to RL on a methodologically broken
starting policy.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import typer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from grpocredit.common.config import load_config  # noqa: E402
from grpocredit.common.logging import init_wandb  # noqa: E402
from grpocredit.common.utils import ensure_dir, write_json  # noqa: E402

log = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, no_args_is_help=True)


def _try_read(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("Failed to read %s: %s", path, e)
        return None


@app.command()
def main(
    sprint_dir: str = typer.Option("experiments/sprint"),
    config: str = typer.Option("configs/base_qwen_math.yaml"),
    run_name: str = typer.Option("sprint-d3-gate"),
    gate_concordance_rho_pass: float = typer.Option(
        0.30, help="ρ(emb_var, reward_var) above this → Stage 2 is a useful VoI proxy"
    ),
    gate_concordance_rho_marginal: float = typer.Option(
        0.15, help="ρ between marginal and pass → proceed with caveats"
    ),
    gate_kappa_pass: float = typer.Option(3.0),
    gate_kappa_marginal: float = typer.Option(2.0),
    gate_group_variance_pass: float = typer.Option(
        0.5,
        help=(
            "§5 of sft_warmup_plan.md: PASS iff fraction of G-groups with "
            "Var(reward) > 0 at step 0 is ≥ this. Below ≈0.5 means the "
            "starting policy is too saturated (e.g., Qwen-Instruct) or too "
            "weak — pivot to a different `π_ref` before RL."
        ),
    ),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sprint_path = Path(sprint_dir)

    cfg = load_config(config, overrides={"output_dir": sprint_dir, "name": run_name})
    wandb_run = init_wandb(cfg, run_name=run_name, extra_config={"sprint_day": 3})

    conc = _try_read(sprint_path / "emb_var_summary.json")
    oracle = _try_read(sprint_path / "oracle_summary.json")
    corrs = _try_read(sprint_path / "oracle_correlations.json")
    gv = _try_read(sprint_path / "day1_group_variance.json")

    concordance_rho = conc["config_a"]["rho_cosine"] if conc and "config_a" in conc else (conc.get("rho_cosine") if conc else None)
    concordance_rho_ci = conc["config_a"].get("rho_cosine_ci") if conc and "config_a" in conc else (conc.get("rho_cosine_ci") if conc else None)
    concordance_rho_b = conc["config_b"]["rho_cosine"] if conc and "config_b" in conc else None
    # Selection metrics (primary concordance signals)
    def _get_conc(key, cfg_key="config_a"):
        if not conc:
            return None
        if cfg_key in conc:
            return conc[cfg_key].get(key)
        return conc.get(key)
    top1_agreement = _get_conc("top1_agreement")
    top1_agreement_b = _get_conc("top1_agreement", "config_b")
    kappa_emb = _get_conc("kappa_emb")
    kappa_emb_b = _get_conc("kappa_emb", "config_b")
    kappa = oracle["kappa"] if oracle else None
    rho_h_fwd = oracle.get("rho_H_fwd") if oracle else None
    rho_h_fwd_ci = oracle.get("rho_H_fwd_ci", [None, None]) if oracle else [None, None]
    rho_h_token = oracle.get("rho_H_token") if oracle else None
    rho_h_token_ci = oracle.get("rho_H_token_ci", [None, None]) if oracle else [None, None]
    h_fwd_k = oracle.get("h_fwd_k") if oracle else None
    # Day 2B selection metrics
    sel_h_fwd_top1 = oracle.get("sel_H_fwd_top1") if oracle else None
    sel_h_fwd_kappa = oracle.get("sel_H_fwd_kappa") if oracle else None
    sel_h_fwd_max_top1 = oracle.get("sel_H_fwd_max_top1") if oracle else None
    sel_h_fwd_max_kappa = oracle.get("sel_H_fwd_max_kappa") if oracle else None
    sel_h_token_top1 = oracle.get("sel_H_token_top1") if oracle else None
    sel_h_token_kappa = oracle.get("sel_H_token_kappa") if oracle else None
    rho_gate = oracle.get("rho_gate") if oracle else None
    position_shape = oracle.get("position_shape") if oracle else None
    gv_frac = gv["fraction_informative"] if gv else None

    # Decisions
    def concordance_decision() -> str:
        # Primary: selection metrics (top-1 agreement and κ_emb)
        # Fallback: Spearman ρ (legacy, if selection metrics not available)
        if top1_agreement is not None and kappa_emb is not None:
            if top1_agreement >= 0.5 and kappa_emb >= 1.5:
                return "pass"
            if top1_agreement >= 0.35 or kappa_emb >= 1.0:
                return "marginal"
            return "fail"
        # Legacy path for old emb_var_summary.json without selection metrics
        if concordance_rho is None:
            return "missing"
        if concordance_rho >= gate_concordance_rho_pass:
            return "pass"
        if concordance_rho >= gate_concordance_rho_marginal:
            return "marginal"
        return "fail"

    def kappa_decision() -> str:
        if kappa is None:
            return "missing"
        if kappa >= gate_kappa_pass:
            return "pass"
        if kappa >= gate_kappa_marginal:
            return "marginal"
        return "fail"

    def rho_decision() -> str:
        # Best available per-trajectory κ_signal from Day 2B
        best_kappa = max(
            (v for v in [sel_h_fwd_kappa, sel_h_fwd_max_kappa, sel_h_token_kappa] if v is not None),
            default=None,
        )
        if best_kappa is None:
            return "missing"
        if best_kappa >= 1.5:
            return "pass"
        if best_kappa >= 1.0:
            return "marginal"
        return "fail"

    def group_variance_decision() -> str:
        # sft_warmup_plan.md §5: this is a hard gate (no marginal band).
        # Below the threshold, the starting policy is methodologically broken
        # for our credit-assignment claim — RL'ing on top of it is not a
        # defensible experiment regardless of how the other gates land.
        if gv_frac is None:
            return "missing"
        if gv_frac >= gate_group_variance_pass:
            return "pass"
        return "fail"

    c_dec = concordance_decision()
    k_dec = kappa_decision()
    r_dec = rho_decision()
    gv_dec = group_variance_decision()

    decisions = {
        "concordance": c_dec,
        "kappa": k_dec,
        "oracle_signal": r_dec,
        "group_variance": gv_dec,
    }
    n_pass = sum(1 for v in decisions.values() if v == "pass")
    n_marginal = sum(1 for v in decisions.values() if v.startswith("marginal"))
    n_fail = sum(1 for v in decisions.values() if v == "fail")

    if gv_dec == "fail":
        # The §5 gate is the most upstream of all — failing it means the
        # starting policy itself is wrong. None of the downstream oracle
        # numbers are interpretable until this is fixed.
        overall = "wrong_starting_policy"
        exit_code = 6
        next_steps = (
            "Group-variance gate FAILED at step 0 — starting policy is too "
            "saturated or too weak (sft_warmup_plan.md §5). DO NOT proceed "
            "to RL. Switch `π_ref` to a SFT-warmed model (Option B: "
            "`realtreetune/deepseekmath-7b-sft-{MATH-v2,GSM8K}`; or Option A: "
            "SFT `Qwen2.5-Math-7B` base ourselves). Re-run sprint Day 1+."
        )
    elif c_dec == "fail":
        overall = "pivot_plan_b"
        exit_code = 4
        next_steps = (
            "Concordance FAILED — lookahead embedding diversity does not predict "
            "terminal reward diversity. Stage 2 clustering is not informative. "
            "Consider: (a) continuous emb_var as VoI signal without clustering, "
            "(b) Plan B — doubly-robust GRPO with IG-as-implicit-baseline (plan §8)."
        )
    elif n_fail > 0:
        overall = "pivot_plan_b"
        exit_code = 4
        next_steps = "Hard fail on a non-concordance gate — discuss before committing to main plan."
    elif n_marginal >= 2:
        overall = "pilot_required"
        exit_code = 3
        next_steps = (
            "2+ gates marginal — run a 200-step pilot training `voi_stage1` vs `grpo`. "
            "If measurable grad-var reduction appears, proceed; else pivot."
        )
    elif n_marginal == 1:
        overall = "proceed_with_caveats"
        exit_code = 2
        next_steps = "1 gate marginal — proceed with main plan, note the caveat in the paper."
    elif n_pass >= 4:
        overall = "proceed"
        exit_code = 0
        next_steps = "All gates pass — launch 15 training runs per experiment plan §3.1."
    else:
        overall = "missing_data"
        exit_code = 5
        next_steps = "Sprint outputs missing — re-run Day 2 before proceeding."

    # Write machine-readable decision
    decision_json = {
        # Selection metrics (primary concordance signals)
        "top1_agreement": top1_agreement,
        "top1_agreement_b": top1_agreement_b,
        "kappa_emb": kappa_emb,
        "kappa_emb_b": kappa_emb_b,
        # Legacy Spearman
        "concordance_rho_cosine": concordance_rho,
        "concordance_rho_ci": concordance_rho_ci,
        "concordance_rho_cosine_b": concordance_rho_b,
        "kappa": kappa,
        "rho_H_fwd": rho_h_fwd,
        "rho_H_fwd_ci": rho_h_fwd_ci,
        "rho_H_token": rho_h_token,
        "rho_H_token_ci": rho_h_token_ci,
        "h_fwd_k": h_fwd_k,
        # Day 2B selection metrics
        "sel_H_fwd_top1": sel_h_fwd_top1,
        "sel_H_fwd_kappa": sel_h_fwd_kappa,
        "sel_H_fwd_max_top1": sel_h_fwd_max_top1,
        "sel_H_fwd_max_kappa": sel_h_fwd_max_kappa,
        "sel_H_token_top1": sel_h_token_top1,
        "sel_H_token_kappa": sel_h_token_kappa,
        "rho_gate": rho_gate,
        "position_shape": position_shape,
        "group_variance_fraction_informative": gv_frac,
        "group_variance_threshold": gate_group_variance_pass,
        "gates": decisions,
        "overall": overall,
        "exit_code": exit_code,
        "next_steps": next_steps,
    }
    write_json(sprint_path / "gate_decision.json", decision_json)

    def _fmt(x: Any, spec: str = ".3f") -> str:
        if x is None:
            return "missing"
        if isinstance(x, float):
            return format(x, spec)
        return str(x)

    # Write GATE_REPORT.md
    report_lines = [
        "# Sprint Gate Report",
        "",
        f"**Overall:** `{overall}` (exit_code={exit_code})",
        "",
        "## Gates",
        "",
        "| Check | Value | Threshold | Status |",
        "|---|---|---|---|",
        f"| Group-variance fraction at step 0 (sft_warmup_plan §5) | {_fmt(gv_frac)} | ≥ {gate_group_variance_pass} | {gv_dec} |",
        f"| Concordance: top-1 agreement (per-traj) | {_fmt(top1_agreement)} | ≥ 0.50 | {c_dec} |",
        f"| Concordance: κ_emb (selection concentration) | {_fmt(kappa_emb)} | ≥ 1.50 | {c_dec} |",
        f"| κ (variance concentration) | {_fmt(kappa)} | ≥ {gate_kappa_pass} | {k_dec} |",
        f"| Oracle signal: best κ_sig (per-traj, H_fwd/H_fwd_max/H_token) | {_fmt(max((v for v in [sel_h_fwd_kappa, sel_h_fwd_max_kappa, sel_h_token_kappa] if v is not None), default=None))} | ≥ 1.50 | {r_dec} |",
        f"| Position curve | `{position_shape or 'missing'}` | mid-peak preferred | — |",
        "",
        "## Next steps",
        "",
        next_steps,
        "",
        "## Raw values",
        "",
        f"- group-variance fraction (informative): {gv_frac}",
        f"- **Concordance selection metrics (config A: K=4/len=30):**",
        f"  - top-1 agreement: {top1_agreement}",
        f"  - κ_emb: {kappa_emb}",
        f"  - ρ(emb_var, reward_var) Spearman (legacy): {concordance_rho}",
        f"  - Spearman 95% CI: {concordance_rho_ci}",
        f"- **Concordance selection metrics (config B: K=8/len=15):**",
        f"  - top-1 agreement: {top1_agreement_b}",
        f"  - κ_emb: {kappa_emb_b}",
        f"  - ρ(emb_var, reward_var) Spearman (legacy): {concordance_rho_b}",
        f"- κ: {kappa}",
        f"- **Day 2B selection metrics (per-trajectory, vs Var(Q^π)):**",
        f"  - H_fwd: top1={sel_h_fwd_top1}, κ_sig={sel_h_fwd_kappa}",
        f"  - H_fwd_max: top1={sel_h_fwd_max_top1}, κ_sig={sel_h_fwd_max_kappa}",
        f"  - H_token: top1={sel_h_token_top1}, κ_sig={sel_h_token_kappa}",
        f"- Spearman correlations (legacy):",
        f"  - ρ(H_fwd, Var(Q^π)): {rho_h_fwd} (K={h_fwd_k})",
        f"  - ρ(H_fwd) 95% CI: [{rho_h_fwd_ci[0]}, {rho_h_fwd_ci[1]}]",
        f"  - ρ(H_token, Var(Q^π)): {rho_h_token}",
        f"  - ρ(H_token) 95% CI: [{rho_h_token_ci[0]}, {rho_h_token_ci[1]}]",
        f"- position shape: {position_shape}",
        "",
    ]
    (sprint_path / "GATE_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")

    # Log to wandb
    wandb_run.log_summary(**decision_json)
    wandb_run.log_table(
        "gate/decisions",
        columns=["check", "value", "threshold", "status"],
        rows=[
            ["group_variance_fraction", gv_frac, gate_group_variance_pass, gv_dec],
            ["concordance_top1", top1_agreement, 0.50, c_dec],
            ["concordance_kappa_emb", kappa_emb, 1.50, c_dec],
            ["kappa", kappa, gate_kappa_pass, k_dec],
            ["oracle_signal_kappa_sig", max((v for v in [sel_h_fwd_kappa, sel_h_fwd_max_kappa, sel_h_token_kappa] if v is not None), default=None), 1.50, r_dec],
        ],
    )
    # position_shape is logged separately (string-valued, can't mix into numeric table)
    wandb_run.log({"gate/position_shape": position_shape or "missing"})
    wandb_run.log_artifact(
        sprint_path / "GATE_REPORT.md",
        artifact_type="gate_report",
        name="GATE_REPORT",
    )
    wandb_run.log_artifact(
        sprint_path / "gate_decision.json",
        artifact_type="gate_report",
        name="gate_decision",
    )

    # ── Human printout ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SPRINT GATE REPORT")
    print("=" * 60)
    for k, v in decision_json.items():
        if k == "gates":
            print("gates:")
            for gk, gv in v.items():
                print(f"  {gk}: {gv}")
        else:
            print(f"{k}: {v}")
    print("=" * 60)
    print(f"OVERALL: {overall}")
    print(f"NEXT: {next_steps}")
    wandb_run.finish(exit_code=exit_code)
    if exit_code != 0:
        raise typer.Exit(exit_code)


if __name__ == "__main__":
    app()
