"""Sprint Day 2B — Q^π-variance oracle + κ + ρ + position curve.

See plan §2B. One oracle run; five deliverables:
  - `oracle_q_variance.json`    (per-boundary Var(Q^π) + cheap signals)
  - `oracle_kappa.txt`          (κ scalar + bootstrap CI)
  - `oracle_correlations.json`  (ρ for H_token, H_fwd, H_sem, s_2, with Fisher-z CI)
  - `oracle_position_curve.csv` (decile curve + shape classification)
  - `position_lookup.csv`       (normalised curve for Stage1 w_pos='lookup')

All logged as wandb artifacts + metrics.

Uses *informative* prompts identified via a group-variance probe (same logic
as Day 2A), so the oracle runs on prompts with genuine outcome diversity rather
than trivially all-correct or all-wrong groups.

H_fwd — average entropy over the next K tokens — replaces cluster-based H_sem
as the primary multi-step signal.  H_sem is still computed for comparison.
"""

from __future__ import annotations

import csv
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
from grpocredit.oracle.group_variance import (  # noqa: E402
    compute_group_variance_report,
    grouped_rewards_from_runner_output,
)
from grpocredit.oracle.kappa_estimator import estimate_kappa  # noqa: E402
from grpocredit.oracle.position_curve import compute_position_curve  # noqa: E402
from grpocredit.oracle.q_variance_oracle import QVarianceOracle  # noqa: E402
from grpocredit.rollout.datasets import load_prompts  # noqa: E402
from grpocredit.rollout.verifier import MathVerifier  # noqa: E402
from grpocredit.voi.stage1_entropy import Stage1Scorer  # noqa: E402

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


def _per_trajectory_selection_metrics(
    records: list,
    signal_name: str,
    signal_field: str,
) -> dict[str, float | int]:
    """Per-trajectory selection metrics for a VoI signal against Var(Q^π).

    Groups oracle records by trajectory_id, then computes:
      (a) top1_agreement — does argmax(signal) == argmax(var_q_pi)?
      (b) kappa_signal — var_q_pi at selected / mean var_q_pi
      (c) overlap@2 — Jaccard of top-2 sets (trajs with ≥4 boundaries)
    """
    from collections import defaultdict

    by_traj: dict[str, list] = defaultdict(list)
    for r in records:
        if r.var_q_pi is not None and not np.isnan(r.var_q_pi):
            by_traj[r.trajectory_id].append(r)

    top1_hits = 0
    top1_total = 0
    kappa_ratios: list[float] = []
    overlap_sum = 0.0
    overlap_total = 0

    for recs in by_traj.values():
        if len(recs) < 2:
            continue

        sig_vals = [getattr(r, signal_field) or 0.0 for r in recs]
        var_vals = [r.var_q_pi for r in recs]

        # (a) top-1 agreement
        top1_total += 1
        best_sig = max(range(len(recs)), key=lambda i: sig_vals[i])
        best_var = max(range(len(recs)), key=lambda i: var_vals[i])
        if best_sig == best_var:
            top1_hits += 1

        # (b) κ_signal
        mean_vq = float(np.mean(var_vals))
        if mean_vq > 1e-12:
            kappa_ratios.append(var_vals[best_sig] / mean_vq)

        # (c) overlap@2
        if len(recs) >= 4:
            overlap_total += 1
            top2_sig = set(sorted(range(len(recs)), key=lambda i: sig_vals[i], reverse=True)[:2])
            top2_var = set(sorted(range(len(recs)), key=lambda i: var_vals[i], reverse=True)[:2])
            overlap_sum += len(top2_sig & top2_var) / len(top2_sig | top2_var)

    return {
        "signal": signal_name,
        "top1_agreement": top1_hits / max(1, top1_total),
        "top1_n_trajectories": top1_total,
        "kappa_signal": float(np.mean(kappa_ratios)) if kappa_ratios else float("nan"),
        "kappa_signal_n_trajectories": len(kappa_ratios),
        "overlap_at_2": overlap_sum / max(1, overlap_total),
        "overlap_at_2_n_trajectories": overlap_total,
    }


@app.command()
def main(
    config: str = typer.Option(..., help="Path to base YAML config"),
    n_prompts: int = typer.Option(256, help="Number of prompts for group-variance probe"),
    group_size: int = typer.Option(8, help="G: rollouts per prompt for informativeness check"),
    h_fwd_k: int = typer.Option(0, help="K for H_fwd (0 = use config oracle.h_fwd_k)"),
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

    effective_h_fwd_k = h_fwd_k if h_fwd_k > 0 else cfg.oracle.h_fwd_k

    # ── 1. Group-variance probe → informative prompts ───────────
    from grpocredit.rollout.vllm_runner import VLLMRolloutRunner
    from scripts._shared import (
        build_prompts,
        detect_all_boundaries,
        results_to_trajectories,
    )

    dataset = cfg.data.train_datasets[0] if cfg.data.train_datasets else "math"
    prompts = load_prompts(dataset, split="train", n=n_prompts)
    runner = VLLMRolloutRunner(cfg.model, cfg.rollout)
    tokenizer = runner.tokenizer
    formatted = build_prompts(prompts, tokenizer, template=cfg.data.prompt_template)
    verifier = MathVerifier()

    log.info("Running group-variance probe: %d prompts × G=%d", n_prompts, group_size)
    gv_groups = runner.generate_from_prompts(
        formatted, n_per_prompt=group_size, seed=cfg.seed + 7
    )
    gt_answers = [pr.ground_truth_answer for pr in prompts]
    grouped_rewards = grouped_rewards_from_runner_output(gv_groups, verifier, gt_answers)
    gv_report = compute_group_variance_report(grouped_rewards)

    log.info(
        "Group-variance probe: %d/%d informative (fraction=%.3f)",
        gv_report.n_informative,
        gv_report.n_groups,
        gv_report.fraction_informative,
    )

    # Filter to informative prompts
    informative_indices: list[int] = []
    for i, rewards in enumerate(grouped_rewards):
        arr = np.asarray(rewards, dtype=float)
        if arr.std() > 1e-12:
            informative_indices.append(i)

    if not informative_indices:
        log.error("No informative prompts found — cannot run oracle")
        wandb_run.finish()
        return

    log.info("Selected %d informative prompts for oracle", len(informative_indices))

    # Pick one trajectory per informative prompt (first rollout from each group)
    informative_prompts = [prompts[i] for i in informative_indices]
    informative_groups = [[gv_groups[i][0]] for i in informative_indices]
    trajectories = results_to_trajectories(informative_prompts, informative_groups, verifier)

    log.info("Created %d trajectories from informative prompts", len(trajectories))

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

    # Stage 1 scores (cheap) — includes H_fwd
    stage1 = Stage1Scorer(cfg.cascade.stage1)
    for t in trajectories:
        stage1.score(
            t,
            boundaries_by_traj.get(t.trajectory_id, []),
            h_fwd_k=effective_h_fwd_k,
        )

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
            "n_informative_prompts": len(informative_indices),
            "n_prompts_total": n_prompts,
            "h_fwd_k": effective_h_fwd_k,
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
            "oracle/n_informative_prompts": len(informative_indices),
            "oracle/h_fwd_k": effective_h_fwd_k,
        }
    )

    # ── 4. Correlations ρ(·, Var(Q^π)) with Fisher-z CI ───────────
    var_q = [r.var_q_pi if r.var_q_pi is not None else np.nan for r in q_result.records]
    signals = {
        "H_token": [r.h_token or 0.0 for r in q_result.records],
        "H_fwd": [r.h_fwd or 0.0 for r in q_result.records],
        "H_fwd_max": [r.h_fwd_max or 0.0 for r in q_result.records],
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

    # ── 4b. Per-trajectory selection metrics ───────────────────────
    sel_signals = [
        ("H_fwd", "h_fwd"),
        ("H_fwd_max", "h_fwd_max"),
        ("H_token", "h_token"),
    ]
    sel_rows: list[dict] = []
    for sig_name, sig_field in sel_signals:
        sel = _per_trajectory_selection_metrics(q_result.records, sig_name, sig_field)
        sel_rows.append(sel)
        wandb_run.log({
            f"oracle/sel_{sig_name}_top1": sel["top1_agreement"],
            f"oracle/sel_{sig_name}_kappa": sel["kappa_signal"],
            f"oracle/sel_{sig_name}_overlap2": sel["overlap_at_2"],
        })
    sel_path = out_dir / "oracle_selection_metrics.json"
    write_json(sel_path, sel_rows)
    wandb_run.log_artifact(sel_path, artifact_type="oracle", name="oracle_selection_metrics")

    log.info("Per-trajectory selection metrics:")
    for sr in sel_rows:
        kappa_str = f"{sr['kappa_signal']:.3f}" if not np.isnan(sr['kappa_signal']) else "NaN"
        log.info(
            "  %s: top1=%.3f (%d trajs)  κ=%-6s (%d trajs)  overlap@2=%.3f (%d trajs)",
            sr["signal"], sr["top1_agreement"], sr["top1_n_trajectories"],
            kappa_str, sr["kappa_signal_n_trajectories"],
            sr["overlap_at_2"], sr["overlap_at_2_n_trajectories"],
        )

    # ── 5. κ ───────────────────────────────────────────────────────
    # Pick the best-performing signal for κ selection: prefer h_fwd_max
    # if it has variance, then h_fwd, then h_token.
    kappa_score = "h_token"
    for candidate in ("h_fwd_max", "h_fwd"):
        vals = [getattr(r, candidate) for r in q_result.records
                if getattr(r, candidate) is not None]
        if len(vals) >= 4 and np.std(vals) > 1e-12:
            kappa_score = candidate
            break
    log.info("Computing κ with bootstrap CI (selection_score=%s)", kappa_score)
    kappa_res = estimate_kappa(
        q_result.records,
        selection_score=kappa_score,
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
            f"h_fwd_k              = {effective_h_fwd_k}\n"
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
            "oracle/kappa_selection_score": kappa_score,
        }
    )

    # ── 6. Position curve ─────────────────────────────────────────
    curve = compute_position_curve(q_result.records)
    pos_csv = out_dir / "oracle_position_curve.csv"
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
    rho_h_fwd = next(r for r in corr_rows if r["signal"] == "H_fwd")
    rho_h_fwd_max = next(r for r in corr_rows if r["signal"] == "H_fwd_max")
    rho_h_token = next(r for r in corr_rows if r["signal"] == "H_token")
    sel_h_fwd = next(s for s in sel_rows if s["signal"] == "H_fwd")
    sel_h_fwd_max = next(s for s in sel_rows if s["signal"] == "H_fwd_max")
    sel_h_token = next(s for s in sel_rows if s["signal"] == "H_token")
    summary = {
        "n_records": len(q_result.records),
        "total_rollouts": q_result.total_rollouts,
        "n_informative_prompts": len(informative_indices),
        "n_prompts_total": n_prompts,
        "h_fwd_k": effective_h_fwd_k,
        "kappa": kappa_res.kappa,
        "kappa_ci": [kappa_res.kappa_ci_low, kappa_res.kappa_ci_high],
        "kappa_selection_score": kappa_score,
        "rho_gate": kappa_res.rho_gate,
        # Selection metrics (primary)
        "selection_metrics": sel_rows,
        "sel_H_fwd_top1": sel_h_fwd["top1_agreement"],
        "sel_H_fwd_kappa": sel_h_fwd["kappa_signal"],
        "sel_H_fwd_max_top1": sel_h_fwd_max["top1_agreement"],
        "sel_H_fwd_max_kappa": sel_h_fwd_max["kappa_signal"],
        "sel_H_token_top1": sel_h_token["top1_agreement"],
        "sel_H_token_kappa": sel_h_token["kappa_signal"],
        # Spearman (legacy)
        "rho_H_fwd": rho_h_fwd["rho"],
        "rho_H_fwd_ci": [rho_h_fwd["ci_low"], rho_h_fwd["ci_high"]],
        "rho_H_fwd_max": rho_h_fwd_max["rho"],
        "rho_H_fwd_max_ci": [rho_h_fwd_max["ci_low"], rho_h_fwd_max["ci_high"]],
        "rho_H_token": rho_h_token["rho"],
        "rho_H_token_ci": [rho_h_token["ci_low"], rho_h_token["ci_high"]],
        "position_shape": curve.shape_classification,
        "coverage_median": float(np.median(coverages)) if coverages else 0.0,
        "tail_stratum_frac": tail_run_frac,
        "group_variance": gv_report.to_dict(),
    }
    write_json(out_dir / "oracle_summary.json", summary)
    wandb_run.log_summary(**{k: v for k, v in summary.items()
                             if k not in ("group_variance", "selection_metrics")})

    print("\nOracle summary")
    print("=" * 55)
    print(f"  informative prompts: {len(informative_indices)}/{n_prompts}")
    print(f"  oracle records:      {len(q_result.records)}")
    print(f"  total rollouts:      {q_result.total_rollouts}")
    print(f"\n  Per-trajectory selection metrics (primary):")
    for sr in sel_rows:
        kappa_str = f"{sr['kappa_signal']:.3f}" if not np.isnan(sr['kappa_signal']) else "NaN"
        print(f"    {sr['signal']:10s}  top1={sr['top1_agreement']:.3f}  "
              f"κ_sig={kappa_str}  overlap@2={sr['overlap_at_2']:.3f}  "
              f"(n_traj={sr['top1_n_trajectories']})")
    print(f"\n  Spearman ρ (global, legacy):")
    for cr in corr_rows:
        print(f"    ρ({cr['signal']:8s}, Var(Q^π)) = {cr['rho']:.3f}  "
              f"95% CI [{cr['ci_low']:.3f}, {cr['ci_high']:.3f}]")
    print(f"\n  κ={kappa_res.kappa:.3f} [{kappa_res.kappa_ci_low:.3f}, "
          f"{kappa_res.kappa_ci_high:.3f}]  (score={kappa_score})")
    print(f"  ρ_gate={kappa_res.rho_gate:.3f}")
    print(f"  position_shape={curve.shape_classification}")
    print(f"  Group variance: {gv_report.n_informative}/{gv_report.n_groups} informative "
          f"(fraction={gv_report.fraction_informative:.3f})")
    wandb_run.finish()


if __name__ == "__main__":
    app()
