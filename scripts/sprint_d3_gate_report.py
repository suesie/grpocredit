"""Sprint Day 3 — gate report.

Reads:
  experiments/sprint/concordance_mi.json
  experiments/sprint/oracle_summary.json
  experiments/sprint/oracle_correlations.json
  experiments/sprint/oracle_kappa.txt

Writes:
  experiments/sprint/GATE_REPORT.md   — human-readable decision table
  experiments/sprint/gate_decision.json — machine-readable

Prints a decision table and exits with:
  0  — all gates pass → proceed with main plan
  2  — marginal (1 gate marginal) → proceed with caveats
  3  — 2+ marginal → pilot training needed
  4  — hard fail on concordance → pivot to Plan B (§8 of plan)

The exit code lets `scripts/main_train.sh` decide whether to launch the main
experiment or fork into Plan B.
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
    gate_concordance_pass_bits: float = typer.Option(0.30),
    gate_concordance_nli_bits: float = typer.Option(0.15),
    gate_kappa_pass: float = typer.Option(3.0),
    gate_kappa_marginal: float = typer.Option(2.0),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sprint_path = Path(sprint_dir)

    cfg = load_config(config, overrides={"output_dir": sprint_dir, "name": run_name})
    wandb_run = init_wandb(cfg, run_name=run_name, extra_config={"sprint_day": 3})

    conc = _try_read(sprint_path / "concordance_mi.json")
    oracle = _try_read(sprint_path / "oracle_summary.json")
    corrs = _try_read(sprint_path / "oracle_correlations.json")

    concordance_mi = conc["mean_mi_bits"] if conc else None
    kappa = oracle["kappa"] if oracle else None
    rho_s2 = oracle["rho_s2"] if oracle else None
    rho_s2_ci = oracle["rho_s2_ci"] if oracle else [None, None]
    rho_gate = oracle["rho_gate"] if oracle else None
    position_shape = oracle["position_shape"] if oracle else None

    # Decisions
    def concordance_decision() -> str:
        if concordance_mi is None:
            return "missing"
        if concordance_mi > gate_concordance_pass_bits:
            return "pass"
        if concordance_mi > gate_concordance_nli_bits:
            return "marginal_nli"
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
        if rho_s2 is None or rho_gate is None or rho_s2_ci[0] is None:
            return "missing"
        if rho_s2_ci[0] >= rho_gate:
            return "pass"
        if rho_s2 >= rho_gate:
            return "marginal"
        return "fail"

    c_dec = concordance_decision()
    k_dec = kappa_decision()
    r_dec = rho_decision()

    decisions = {"concordance": c_dec, "kappa": k_dec, "rho_s2": r_dec}
    n_pass = sum(1 for v in decisions.values() if v == "pass")
    n_marginal = sum(1 for v in decisions.values() if v.startswith("marginal"))
    n_fail = sum(1 for v in decisions.values() if v == "fail")

    if c_dec == "fail":
        overall = "pivot_plan_b"
        exit_code = 4
        next_steps = "Pivot to Plan B — doubly-robust GRPO with IG-as-implicit-baseline (plan §8)."
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
    elif n_pass >= 3:
        overall = "proceed"
        exit_code = 0
        next_steps = "All gates pass — launch 15 training runs per experiment plan §3.1."
    else:
        overall = "missing_data"
        exit_code = 5
        next_steps = "Sprint outputs missing — re-run Day 2 before proceeding."

    # Write machine-readable decision
    decision_json = {
        "concordance_mi_bits": concordance_mi,
        "kappa": kappa,
        "rho_s2": rho_s2,
        "rho_s2_ci": rho_s2_ci,
        "rho_gate": rho_gate,
        "position_shape": position_shape,
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
        f"| Concordance MI | {_fmt(concordance_mi)} bits | > {gate_concordance_pass_bits} bits | {c_dec} |",
        f"| κ (variance concentration) | {_fmt(kappa)} | ≥ {gate_kappa_pass} | {k_dec} |",
        f"| ρ(s_2, Var(Q^π)) 95% CI low | {_fmt(rho_s2_ci[0])} | ≥ ρ_gate = {_fmt(rho_gate)} | {r_dec} |",
        f"| Position curve | `{position_shape or 'missing'}` | mid-peak preferred | — |",
        "",
        "## Next steps",
        "",
        next_steps,
        "",
        "## Raw values",
        "",
        f"- concordance mean MI (bits): {concordance_mi}",
        f"- κ: {kappa}",
        f"- ρ(s_2, Var(Q^π)): {rho_s2}",
        f"- ρ 95% CI: [{rho_s2_ci[0]}, {rho_s2_ci[1]}]",
        f"- ρ_gate: {rho_gate}",
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
            ["concordance_mi_bits", concordance_mi, gate_concordance_pass_bits, c_dec],
            ["kappa", kappa, gate_kappa_pass, k_dec],
            ["rho_s2_ci_low", rho_s2_ci[0], rho_gate, r_dec],
            ["position_shape", position_shape, "mid_peak", "—"],
        ],
    )
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
