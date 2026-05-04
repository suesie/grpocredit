"""Sprint Day 2A — embedding-variance vs reward-variance diagnostic.

Replaces the old MI(C_prefix, C_terminal) concordance check with a direct
test of whether short-lookahead embedding diversity predicts terminal reward
diversity, which is the core assumption behind Stage 2's VoI proxy.

Steps:
  1. Load N prompts, generate G=8 rollouts per prompt, score with verifier.
  2. Identify *informative* prompts (groups with 0 < mean_reward < 1).
  3. For each informative prompt, pick one trajectory, detect boundaries.
  4. At each boundary, run K=4 terminal continuations.
  5. Compute embedding variance and terminal reward variance.
  6. Report per-trajectory selection metrics:
     - top-1 agreement (does max emb_var == max reward_var within each traj?)
     - overlap@2 (Jaccard of top-2 sets, for trajs with ≥4 boundaries)
     - κ_emb (reward_var at best emb_var bd / mean reward_var in traj)
     Plus legacy Spearman ρ(emb_var, reward_var) across all boundaries.

Gate thresholds (proposed):
    top-1 agreement > 0.5  → embedding signal picks the right boundary
                               more often than random.
    κ_emb > 1.5            → selected boundary has 50%+ higher reward
                               variance than average.

Outputs (under --output-dir):
    emb_var_summary.json
    emb_var_per_boundary.jsonl
    emb_var_per_position.csv

All logged as wandb artifacts + metrics.
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
    seed_everything,
    write_json,
    write_jsonl,
)
from grpocredit.oracle.concordance_check import run_embedding_variance_check  # noqa: E402
from grpocredit.oracle.group_variance import (  # noqa: E402
    compute_group_variance_report,
    grouped_rewards_from_runner_output,
)
from grpocredit.rollout.datasets import load_prompts  # noqa: E402
from grpocredit.rollout.verifier import MathVerifier  # noqa: E402

log = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def main(
    config: str = typer.Option(..., help="Path to base YAML config"),
    n_prompts: int = typer.Option(256, help="Number of prompts for group-variance probe"),
    group_size: int = typer.Option(8, help="G: rollouts per prompt for informativeness check"),
    boundaries_per_trajectory: int = typer.Option(
        8, help="Max boundaries sampled per trajectory"
    ),
    n_continuations: int = typer.Option(4, help="K: continuations per boundary"),
    output_dir: str = typer.Option("experiments/sprint"),
    run_name: str = typer.Option("sprint-d2-concordance"),
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config(config, overrides={"output_dir": output_dir, "name": run_name})
    seed_everything(cfg.seed)
    out_dir = ensure_dir(output_dir)

    wandb_run = init_wandb(
        cfg, run_name=run_name, extra_config={"sprint_day": 2, "step": "2A"}
    )

    # ── 1. Load prompts ────────────────────────────────────────────
    from grpocredit.rollout.vllm_runner import VLLMRolloutRunner
    from scripts._shared import build_prompts, detect_all_boundaries, results_to_trajectories

    dataset = cfg.data.train_datasets[0] if cfg.data.train_datasets else "math"
    prompts = load_prompts(dataset, split="test", n=n_prompts)
    runner = VLLMRolloutRunner(cfg.model, cfg.rollout)
    tokenizer = runner.tokenizer
    formatted = build_prompts(prompts, tokenizer, template=cfg.data.prompt_template)
    verifier = MathVerifier()

    # ── 2. Group-variance probe → identify informative prompts ─────
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

    # Filter to informative prompts (where not all G rollouts have the same reward)
    informative_indices: list[int] = []
    for i, rewards in enumerate(grouped_rewards):
        arr = np.asarray(rewards, dtype=float)
        if arr.std() > 1e-12:
            informative_indices.append(i)

    if not informative_indices:
        log.error("No informative prompts found — cannot run diagnostic")
        wandb_run.finish()
        return

    log.info("Selected %d informative prompts for diagnostic", len(informative_indices))

    # ── 3. Pick one trajectory per informative prompt ──────────────
    # Use the first rollout from each informative group.
    informative_prompts = [prompts[i] for i in informative_indices]
    informative_groups = [[gv_groups[i][0]] for i in informative_indices]
    trajectories = results_to_trajectories(informative_prompts, informative_groups, verifier)

    log.info("Created %d trajectories from informative prompts", len(trajectories))

    # ── 4. Boundary detection + sampling ───────────────────────────
    boundaries_by_traj = detect_all_boundaries(trajectories, tokenizer, cfg.boundary)
    tb_pairs: list[tuple] = []
    for t in trajectories:
        bds = boundaries_by_traj.get(t.trajectory_id, [])
        if not bds:
            continue
        step = max(1, len(bds) // boundaries_per_trajectory)
        selected = bds[::step][:boundaries_per_trajectory]
        for b in selected:
            tb_pairs.append((t, b))

    if not tb_pairs:
        log.error("No boundaries detected in informative trajectories")
        wandb_run.finish()
        return

    def tb_pairs_for_traj(pairs, traj_id):
        return [b for _, b in pairs if _.trajectory_id == traj_id]

    boundary_counts = [len(boundaries_by_traj.get(t.trajectory_id, [])) for t in trajectories]
    mean_boundaries = float(np.mean(boundary_counts)) if boundary_counts else 0.0
    log.info(
        "Sampled %d boundaries across %d trajectories (mean %.1f boundaries/traj)",
        len(tb_pairs),
        len(trajectories),
        mean_boundaries,
    )

    # ── 4b. Save boundary-annotated responses for inspection ───────
    annotated_path = out_dir / "boundary_annotated_responses.txt"
    with annotated_path.open("w", encoding="utf-8") as af:
        for t in trajectories:
            bds = boundaries_by_traj.get(t.trajectory_id, [])
            sampled_positions = set()
            for b in tb_pairs_for_traj(tb_pairs, t.trajectory_id):
                sampled_positions.add(b.token_position)

            # Insert markers into response text at boundary char positions,
            # working right-to-left so earlier positions stay valid.
            text = t.response_text
            for b in sorted(bds, key=lambda b: b.char_position, reverse=True):
                sampled_tag = "*" if b.token_position in sampled_positions else ""
                marker = f"«B{b.boundary_idx}:{b.kind}{sampled_tag}»"
                text = text[: b.char_position] + marker + text[b.char_position :]

            af.write(f"═══ {t.trajectory_id} | reward={t.reward} | "
                     f"tokens={t.length} | boundaries={len(bds)} | "
                     f"prompt_id={t.prompt_id}\n")
            af.write(f"Q: {t.prompt.split('Problem:')[-1].strip()[:200]}\n")
            af.write(f"A (ground truth): {t.ground_truth_answer}\n")
            af.write(f"Response (annotated):\n{text}\n\n")
    log.info("Saved boundary-annotated responses to %s", annotated_path)

    # ── 5. Run embedding-variance diagnostic ───────────────────────
    log.info(
        "Running embedding-variance diagnostic (config A): %d boundaries × K=%d × len=%d",
        len(tb_pairs),
        n_continuations,
        cfg.cascade.stage2.lookahead_max_new_tokens,
    )
    max_new_tokens_terminal = min(cfg.rollout.max_new_tokens, 512)
    result = run_embedding_variance_check(
        backend=runner,
        trajectory_boundaries=tb_pairs,
        verifier=verifier,
        oracle_config=cfg.oracle,
        stage2_config=cfg.cascade.stage2,
        n_continuations=n_continuations,
        max_new_tokens=max_new_tokens_terminal,
        seed=cfg.seed + 7,
        n_informative_prompts=len(informative_indices),
    )

    # ── 5b. Second config: K=8 rollouts × 15-token lookaheads ─────
    #    Reward variance still from 4 rollouts (simulates: 8 cheap previews
    #    to decide, 4 expensive full rollouts to measure).
    log.info(
        "Running embedding-variance diagnostic (config B): %d boundaries × K=8 × len=15, reward from 4",
        len(tb_pairs),
    )
    result_b = run_embedding_variance_check(
        backend=runner,
        trajectory_boundaries=tb_pairs,
        verifier=verifier,
        oracle_config=cfg.oracle,
        stage2_config=cfg.cascade.stage2,
        n_continuations=8,
        n_reward_samples=4,
        lookahead_max_tokens=15,
        max_new_tokens=max_new_tokens_terminal,
        seed=cfg.seed + 11,
        n_informative_prompts=len(informative_indices),
    )

    # ── 6. Outputs ─────────────────────────────────────────────────
    def _result_to_summary(res, label="A"):
        return {
            "config": label,
            "n_emb_samples": res.n_emb_samples,
            "n_reward_samples": res.n_reward_samples,
            "lookahead_max_tokens": res.lookahead_max_tokens,
            # Selection metrics (primary)
            "top1_agreement": res.top1_agreement,
            "top1_n_trajectories": res.top1_n_trajectories,
            "overlap_at_2": res.overlap_at_2,
            "overlap_at_2_n_trajectories": res.overlap_at_2_n_trajectories,
            "kappa_emb": res.kappa_emb,
            "kappa_emb_n_trajectories": res.kappa_emb_n_trajectories,
            # Spearman (legacy)
            "rho_cosine": res.rho_cosine,
            "rho_cosine_ci": list(res.rho_cosine_ci),
            "rho_cosine_long_only": res.rho_cosine_long_only,
            "n_long_boundaries": res.n_long_boundaries,
            "n_informative_prompts": res.n_informative_prompts,
            "n_boundaries": res.n_boundaries,
            "total_rollouts": res.total_rollouts,
            "mean_remaining_tokens": res.mean_remaining_tokens,
            "mean_boundaries_per_traj": mean_boundaries,
        }

    summary = {
        "config_a": _result_to_summary(result, "A: K=4/len=30"),
        "config_b": _result_to_summary(result_b, "B: K=8/len=15/rew=4"),
        "group_variance": gv_report.to_dict(),
    }

    json_path = out_dir / "emb_var_summary.json"
    write_json(json_path, summary)
    wandb_run.log_artifact(json_path, artifact_type="concordance", name="emb_var_summary")

    # Per-boundary records
    raw_path = out_dir / "emb_var_per_boundary.jsonl"
    write_jsonl(raw_path, (r.to_dict() for r in result.records))
    wandb_run.log_artifact(raw_path, artifact_type="concordance", name="emb_var_per_boundary")

    # Per-position-decile breakdown
    csv_path = out_dir / "emb_var_per_position.csv"
    if result.records:
        deciles = np.digitize(
            [r.relative_position for r in result.records], np.linspace(0, 1, 11)
        ) - 1
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "decile", "mean_emb_var_cosine", "mean_emb_var_trace",
                "mean_reward_var", "mean_remaining_tokens", "n_boundaries",
            ])
            for d in range(10):
                mask = deciles == d
                recs_d = [result.records[i] for i in range(len(result.records)) if mask[i]]
                if not recs_d:
                    w.writerow([d, 0.0, 0.0, 0.0, 0.0, 0])
                    continue
                w.writerow([
                    d,
                    f"{np.mean([r.emb_var_cosine for r in recs_d]):.6f}",
                    f"{np.mean([r.emb_var_trace for r in recs_d]):.6f}",
                    f"{np.mean([r.reward_var for r in recs_d]):.6f}",
                    f"{np.mean([r.remaining_tokens for r in recs_d]):.1f}",
                    len(recs_d),
                ])
        wandb_run.log_artifact(csv_path, artifact_type="concordance", name="emb_var_per_position")

    # wandb metrics
    for tag, res in [("a", result), ("b", result_b)]:
        wandb_run.log({
            # Selection metrics (primary)
            f"concordance_{tag}/top1_agreement": res.top1_agreement,
            f"concordance_{tag}/top1_n_trajectories": res.top1_n_trajectories,
            f"concordance_{tag}/overlap_at_2": res.overlap_at_2,
            f"concordance_{tag}/kappa_emb": res.kappa_emb,
            # Spearman (legacy)
            f"concordance_{tag}/rho_cosine": res.rho_cosine,
            f"concordance_{tag}/rho_cosine_ci_low": res.rho_cosine_ci[0],
            f"concordance_{tag}/rho_cosine_ci_high": res.rho_cosine_ci[1],
            f"concordance_{tag}/rho_cosine_long_only": res.rho_cosine_long_only,
            f"concordance_{tag}/n_long_boundaries": res.n_long_boundaries,
            f"concordance_{tag}/n_boundaries": res.n_boundaries,
            f"concordance_{tag}/total_rollouts": res.total_rollouts,
            f"concordance_{tag}/mean_remaining_tokens": res.mean_remaining_tokens,
            f"concordance_{tag}/lookahead_max_tokens": res.lookahead_max_tokens,
            f"concordance_{tag}/n_emb_samples": res.n_emb_samples,
            f"concordance_{tag}/n_reward_samples": res.n_reward_samples,
        })
    # Keep top-level keys for gate report compatibility (config A is the default)
    wandb_run.log({
        "concordance/rho_cosine": result.rho_cosine,
        "concordance/top1_agreement": result.top1_agreement,
        "concordance/kappa_emb": result.kappa_emb,
        "concordance/n_informative_prompts": result.n_informative_prompts,
        "concordance/mean_boundaries_per_traj": mean_boundaries,
    })
    wandb_run.log_summary(
        top1_agreement_a=result.top1_agreement,
        kappa_emb_a=result.kappa_emb,
        rho_cosine_a=result.rho_cosine,
        rho_cosine_long_a=result.rho_cosine_long_only,
        top1_agreement_b=result_b.top1_agreement,
        kappa_emb_b=result_b.kappa_emb,
        rho_cosine_b=result_b.rho_cosine,
        rho_cosine_long_b=result_b.rho_cosine_long_only,
        n_informative_prompts=result.n_informative_prompts,
        n_boundaries=result.n_boundaries,
    )

    # ── 7. Console summary ─────────────────────────────────────────
    def _print_config(label, res):
        print(f"\n  [{label}]  K_emb={res.n_emb_samples}  K_rew={res.n_reward_samples}  "
              f"lookahead={res.lookahead_max_tokens} tokens")
        print(f"    Selection metrics (per-trajectory):")
        print(f"      top-1 agreement:   {res.top1_agreement:.3f}  "
              f"({res.top1_n_trajectories} trajs with ≥2 boundaries)")
        print(f"      overlap@2:         {res.overlap_at_2:.3f}  "
              f"({res.overlap_at_2_n_trajectories} trajs with ≥4 boundaries)")
        kappa_str = f"{res.kappa_emb:.3f}" if not np.isnan(res.kappa_emb) else "NaN"
        print(f"      κ_emb:             {kappa_str}  "
              f"({res.kappa_emb_n_trajectories} trajs with reward_var > 0)")
        print(f"    Spearman (global, legacy):")
        print(f"      ρ(emb_var, reward_var) = {res.rho_cosine:.3f}  "
              f"95% CI [{res.rho_cosine_ci[0]:.3f}, {res.rho_cosine_ci[1]:.3f}]")
        if res.n_long_boundaries >= 4:
            print(f"      Long-only (remaining > {2 * res.lookahead_max_tokens}): "
                  f"n={res.n_long_boundaries}  ρ={res.rho_cosine_long_only:.3f}")
        else:
            print(f"      Long-only: too few boundaries ({res.n_long_boundaries})")

    print("\nEmbedding-variance diagnostic summary")
    print("-" * 50)
    print(f"  informative prompts:   {result.n_informative_prompts}/{n_prompts}")
    print(f"  boundaries analysed:   {result.n_boundaries}")
    print(f"  mean remaining tokens: {result.mean_remaining_tokens:.1f}")
    _print_config("Config A: K=4 / len=30", result)
    _print_config("Config B: K=8 / len=15 / rew=4", result_b)
    print()

    wandb_run.finish()


if __name__ == "__main__":
    app()
