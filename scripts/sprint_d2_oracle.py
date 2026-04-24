"""Sprint Day 2B — Q^π-variance oracle + κ + ρ + position curve.

See plan §2B. One oracle run; three deliverables:
  - `oracle_q_variance.json`    (per-boundary Var(Q^π) + cheap signals)
  - `oracle_kappa.txt`          (κ scalar + bootstrap CI)
  - `oracle_correlations.json`  (ρ for H_token, H_sem, s_2, with Fisher-z CI)
  - `oracle_position_curve.csv` (decile curve + shape classification)
  - `position_lookup.csv`       (normalised curve for Stage1 w_pos='lookup')

All logged as wandb artifacts + metrics.

Cost estimate (plan §2B): 100 trajectories × 5 boundaries × 6 actions × 32
rollouts = 96K rollouts; +12K if tail stratum triggers.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import typer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from grpocredit.common.config import load_config  # noqa: E402
from grpocredit.common.logging import init_wandb  # noqa: E402
from grpocredit.common.utils import (  # noqa: E402
    ensure_dir,
    fisher_z_ci,
    seed_everything,
    write_json,
    write_jsonl,
)
from grpocredit.oracle.kappa_estimator import estimate_kappa  # noqa: E402
from grpocredit.oracle.position_curve import compute_position_curve  # noqa: E402
from grpocredit.oracle.q_variance_oracle import QVarianceOracle  # noqa: E402
from grpocredit.rollout.datasets import load_prompts  # noqa: E402
from grpocredit.rollout.verifier import MathVerifier  # noqa: E402
from grpocredit.voi.stage1_entropy import Stage1Scorer  # noqa: E402
from grpocredit.voi.stage2_semantic import Stage2Scorer  # noqa: E402

log = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, no_args_is_help=True)


def _spearman(x: list[float], y: list[float]) -> float:
    from scipy import stats

    arr_x = np.asarray(x, dtype=float)
    arr_y = np.asarray(y, dtype=float)
    mask = np.isfinite(arr_x) & np.isfinite(arr_y)
    if mask.sum() < 4:
        return float("nan")
    return float(stats.spearmanr(arr_x[mask], arr_y[mask]).correlation)


@app.command()
def main(
    config: str = typer.Option(..., help="Path to base YAML config"),
    output_dir: str = typer.Option("experiments/sprint"),
    run_name: str = typer.Option("sprint-d2-oracle"),
    f_sel: float = typer.Option(0.15, help="Fraction of boundaries κ-selected"),
    f_target: float = typer.Option(0.10, help="Target trajectory-level grad-var reduction"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(config, overrides={"output_dir": output_dir, "name": run_name})
    seed_everything(cfg.seed)
    out_dir = ensure_dir(output_dir)
    wandb_run = init_wandb(cfg, run_name=run_name, extra_config={"sprint_day": 2, "step": "2B"})

    # ── 1. Base trajectories ───────────────────────────────────────
    from grpocredit.rollout.vllm_runner import VLLMRolloutRunner
    from scripts._shared import (
        build_prompts,
        detect_all_boundaries,
        results_to_trajectories,
    )

    prompts = load_prompts(
        cfg.data.train_datasets[0] if cfg.data.train_datasets else "math",
        split="train",
        n=cfg.oracle.n_trajectories,
    )
    runner = VLLMRolloutRunner(cfg.model, cfg.rollout)
    tokenizer = runner.tokenizer
    formatted = build_prompts(prompts, tokenizer, template=cfg.data.prompt_template)
    groups = runner.generate_from_prompts(
        formatted, n_per_prompt=1, seed=cfg.seed
    )
    verifier = MathVerifier()
    trajectories = results_to_trajectories(prompts, groups, verifier)

    # ── 2. Boundaries + Stage 1/2 scores ──────────────────────────
    boundaries_by_traj = detect_all_boundaries(trajectories, tokenizer, cfg.boundary)

    # Sample `boundaries_per_trajectory` from each trajectory's boundary list.
    tbg: list[tuple] = []
    for t in trajectories:
        bds = boundaries_by_traj.get(t.trajectory_id, [])
        if not bds:
            continue
        step = max(1, len(bds) // cfg.oracle.boundaries_per_trajectory)
        selected = bds[::step][: cfg.oracle.boundaries_per_trajectory]
        for b in selected:
            tbg.append((t, b, t.ground_truth_answer))

    log.info("Sampled %d (trajectory, boundary) oracle targets", len(tbg))

    # Stage 1 scores (cheap)
    stage1 = Stage1Scorer(cfg.cascade.stage1)
    for t in trajectories:
        stage1.score(t, boundaries_by_traj.get(t.trajectory_id, []))

    # Stage 2 lookaheads + scores (mid-cost)
    stage2 = Stage2Scorer(cfg.cascade.stage2)
    flat_bds = [b for _, b, _ in tbg]
    flat_trajs = [t for t, _, _ in tbg]
    if flat_bds:
        prefixes = [
            t.prompt_token_ids + t.token_ids[: b.token_position]
            for t, b in zip(flat_trajs, flat_bds, strict=False)
        ]
        la = runner.continue_from_prefixes(
            prefix_token_ids=prefixes,
            n_continuations=cfg.cascade.stage2.n_lookaheads,
            max_new_tokens=cfg.cascade.stage2.lookahead_max_new_tokens,
            temperature=cfg.cascade.stage2.lookahead_temperature,
            top_p=0.95,
            seed=cfg.seed + 23,
        )
        la_map = {id(b): rs for b, rs in zip(flat_bds, la, strict=False)}
        stage2.score(flat_bds, la_map)

    # ── 3. Q^π-variance oracle ────────────────────────────────────
    log.info(
        "Running Q^π-variance oracle: %d boundaries × %d actions × %d rollouts",
        len(tbg),
        cfg.oracle.top_m_actions,
        cfg.oracle.rollouts_per_forced_action,
    )
    oracle = QVarianceOracle(config=cfg.oracle, verifier=verifier)
    q_result = oracle.run(
        backend=runner,
        trajectory_boundaries=tbg,
        max_new_tokens=cfg.rollout.max_new_tokens,
        temperature=cfg.rollout.temperature,
        seed=cfg.seed + 31,
    )
    log.info("Oracle run done — %d rollouts total", q_result.total_rollouts)

    # Per-boundary records to disk
    q_records_path = out_dir / "oracle_q_variance.json"
    write_json(
        q_records_path,
        {
            "n_records": len(q_result.records),
            "total_rollouts": q_result.total_rollouts,
            "config": cfg.oracle.model_dump(),
            "records": [r.to_dict() for r in q_result.records],
        },
    )
    wandb_run.log_artifact(q_records_path, artifact_type="oracle", name="oracle_q_variance")

    # Coverage + tail-stratum usage report
    coverages = [r.coverage_c for r in q_result.records]
    tail_run_frac = sum(1 for r in q_result.records if r.tail_result is not None) / max(
        1, len(q_result.records)
    )
    wandb_run.log(
        {
            "oracle/coverage_mean": float(np.mean(coverages)) if coverages else 0.0,
            "oracle/coverage_median": float(np.median(coverages)) if coverages else 0.0,
            "oracle/tail_stratum_frac": tail_run_frac,
            "oracle/total_rollouts": q_result.total_rollouts,
            "oracle/n_records": len(q_result.records),
        }
    )

    # ── 4. Correlations ρ(·, Var(Q^π)) with Fisher-z CI ───────────
    var_q = [r.var_q_pi if r.var_q_pi is not None else np.nan for r in q_result.records]
    signals = {
        "H_token": [r.h_token or 0.0 for r in q_result.records],
        "H_sem": [r.h_sem or 0.0 for r in q_result.records],
        "s2": [r.s2 or 0.0 for r in q_result.records],
    }
    corr_rows: list[dict] = []
    valid_n = sum(1 for v in var_q if not np.isnan(v))
    for name, sig in signals.items():
        rho = _spearman(sig, var_q)
        lo, hi = fisher_z_ci(rho, valid_n) if valid_n > 4 else (float("nan"), float("nan"))
        corr_rows.append({"signal": name, "rho": rho, "ci_low": lo, "ci_high": hi, "n": valid_n})
        wandb_run.log(
            {
                f"oracle/rho_{name}_vs_Var_Q": rho,
                f"oracle/rho_{name}_ci_low": lo,
                f"oracle/rho_{name}_ci_high": hi,
            }
        )
    corr_path = out_dir / "oracle_correlations.json"
    write_json(corr_path, corr_rows)
    wandb_run.log_artifact(corr_path, artifact_type="oracle", name="oracle_correlations")

    # ── 5. κ ───────────────────────────────────────────────────────
    log.info("Computing κ with bootstrap CI")
    kappa_res = estimate_kappa(
        q_result.records,
        selection_score="s2",
        f_sel=f_sel,
        f_target=f_target,
        seed=cfg.seed + 47,
    )
    (out_dir / "oracle_kappa.txt").write_text(
        (
            f"kappa                = {kappa_res.kappa:.4f}\n"
            f"kappa_ci_95%         = [{kappa_res.kappa_ci_low:.4f}, {kappa_res.kappa_ci_high:.4f}]\n"
            f"mean_grad_var_sel    = {kappa_res.mean_grad_var_selected:.6f}\n"
            f"mean_grad_var_all    = {kappa_res.mean_grad_var_all:.6f}\n"
            f"f_sel                = {kappa_res.f_sel}\n"
            f"rho_gate             = {kappa_res.rho_gate:.4f}\n"
            f"selection_score      = {kappa_res.selection_score}\n"
        ),
        encoding="utf-8",
    )
    wandb_run.log(
        {
            "oracle/kappa": kappa_res.kappa,
            "oracle/kappa_ci_low": kappa_res.kappa_ci_low,
            "oracle/kappa_ci_high": kappa_res.kappa_ci_high,
            "oracle/rho_gate": kappa_res.rho_gate,
            "oracle/mean_grad_var_selected": kappa_res.mean_grad_var_selected,
            "oracle/mean_grad_var_all": kappa_res.mean_grad_var_all,
        }
    )

    # ── 6. Position curve ─────────────────────────────────────────
    curve = compute_position_curve(q_result.records)
    pos_csv = out_dir / "oracle_position_curve.csv"
    import csv

    with pos_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["decile_midpoint", "mean_var", "std_var", "n"])
        for d, m, s, n in zip(
            curve.decile_midpoints,
            curve.mean_var,
            curve.std_var,
            curve.n_per_decile,
            strict=False,
        ):
            w.writerow([f"{d:.3f}", f"{m:.6f}", f"{s:.6f}", n])
    wandb_run.log_artifact(pos_csv, artifact_type="oracle", name="oracle_position_curve")

    lookup_csv = out_dir / "position_lookup.csv"
    curve.to_lookup_csv(lookup_csv)
    wandb_run.log_artifact(lookup_csv, artifact_type="oracle", name="position_lookup")
    wandb_run.log({"oracle/position_shape": curve.shape_classification})
    wandb_run.log_table(
        "oracle/position_curve",
        columns=["decile_midpoint", "mean_var", "std_var", "n"],
        rows=[
            [d, m, s, n]
            for d, m, s, n in zip(
                curve.decile_midpoints,
                curve.mean_var,
                curve.std_var,
                curve.n_per_decile,
                strict=False,
            )
        ],
    )

    # ── 7. Summary ────────────────────────────────────────────────
    rho_s2 = next(r for r in corr_rows if r["signal"] == "s2")
    summary = {
        "n_records": len(q_result.records),
        "total_rollouts": q_result.total_rollouts,
        "kappa": kappa_res.kappa,
        "kappa_ci": [kappa_res.kappa_ci_low, kappa_res.kappa_ci_high],
        "rho_gate": kappa_res.rho_gate,
        "rho_s2": rho_s2["rho"],
        "rho_s2_ci": [rho_s2["ci_low"], rho_s2["ci_high"]],
        "rho_s2_gate_pass": (
            (not np.isnan(rho_s2["ci_low"])) and rho_s2["ci_low"] >= kappa_res.rho_gate
        ),
        "position_shape": curve.shape_classification,
        "coverage_median": float(np.median(coverages)) if coverages else 0.0,
        "tail_stratum_frac": tail_run_frac,
    }
    write_json(out_dir / "oracle_summary.json", summary)
    wandb_run.log_summary(**summary)

    print("\nOracle summary")
    print("-" * 40)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    wandb_run.finish()


if __name__ == "__main__":
    app()
