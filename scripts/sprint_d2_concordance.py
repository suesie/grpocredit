"""Sprint Day 2A — concordance check (RC-F1, plan §2A).

Samples 100 trajectories, detects boundaries, runs K_LA=4 long-terminal
rollouts per boundary, computes per-boundary MI between lookahead-cluster
labels and terminal-cluster labels, then aggregates to mean MI + per-position
deciles.

Gate (plan §2A):
    > 0.3 bits → proceed with Stage 2 cosine clustering.
    0.15–0.30 bits → upgrade to NLI clustering and re-check.
    ≤ 0.15 bits → Plan B (§8 of plan).

Outputs (under --output-dir):
    concordance_mi.json
    concordance_per_position.csv
    concordance_raw.jsonl (per-boundary lookahead/terminal text + labels)

All logged as wandb artifacts + metrics.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

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
from grpocredit.oracle.concordance_check import run_concordance_check  # noqa: E402
from grpocredit.rollout.datasets import load_prompts  # noqa: E402
from grpocredit.rollout.verifier import MathVerifier  # noqa: E402

log = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def main(
    config: str = typer.Option(..., help="Path to base YAML config"),
    n_trajectories: int = typer.Option(100),
    boundaries_per_trajectory: int = typer.Option(
        8, help="Target boundaries sampled per trajectory after detection"
    ),
    output_dir: str = typer.Option("experiments/sprint"),
    run_name: str = typer.Option("sprint-d2-concordance"),
    gate_pass_bits: float = typer.Option(0.30),
    gate_nli_bits: float = typer.Option(0.15),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(config, overrides={"output_dir": output_dir, "name": run_name})
    seed_everything(cfg.seed)
    out_dir = ensure_dir(output_dir)

    wandb_run = init_wandb(cfg, run_name=run_name, extra_config={"sprint_day": 2, "step": "2A"})

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
        n=n_trajectories,
    )
    runner = VLLMRolloutRunner(cfg.model, cfg.rollout)
    tokenizer = runner.tokenizer
    formatted = build_prompts(prompts, tokenizer, template=cfg.data.prompt_template)
    groups = runner.generate_from_prompts(
        formatted, n_per_prompt=1, seed=cfg.seed
    )
    verifier = MathVerifier()
    trajectories = results_to_trajectories(prompts, groups, verifier)

    # ── 2. Boundary detection + sampling ───────────────────────────
    boundaries_by_traj = detect_all_boundaries(trajectories, tokenizer, cfg.boundary)
    tb_pairs: list[tuple] = []
    for t in trajectories:
        bds = boundaries_by_traj.get(t.trajectory_id, [])
        if not bds:
            continue
        # Evenly sample up to `boundaries_per_trajectory` from the detected set
        step = max(1, len(bds) // boundaries_per_trajectory)
        selected = bds[::step][:boundaries_per_trajectory]
        for b in selected:
            tb_pairs.append((t, b))

    log.info(
        "Sampled %d boundaries across %d trajectories for concordance",
        len(tb_pairs),
        len(trajectories),
    )
    if len(tb_pairs) < cfg.oracle.concordance_min_boundaries:
        log.warning(
            "Only %d boundaries sampled (< %d target) — MI estimate will be noisy",
            len(tb_pairs),
            cfg.oracle.concordance_min_boundaries,
        )

    # ── 3. Run concordance (one batched vLLM call + clustering) ────
    log.info("Running concordance check (terminal rollouts + sentence-T5 clustering)")
    # Cap terminal rollouts at 512 tokens to bound the Day-2A cost (Stage-2 lookahead
    # needs only 30 tokens; the terminal extension after that is typically ≤300 tokens
    # on the MATH distribution).
    max_new_tokens_terminal = min(cfg.rollout.max_new_tokens, 512)

    result = run_concordance_check(
        backend=runner,
        trajectory_boundaries=tb_pairs,
        oracle_config=cfg.oracle,
        stage2_config=cfg.cascade.stage2,
        max_new_tokens=max_new_tokens_terminal,
        seed=cfg.seed + 7,
    )

    # ── 4. Outputs ─────────────────────────────────────────────────
    json_path = out_dir / "concordance_mi.json"
    write_json(
        json_path,
        {
            "mean_mi_bits": result.mean_mi_bits,
            "median_mi_bits": result.median_mi_bits,
            "n_boundaries": len(result.per_boundary),
            "n_trajectories": len(trajectories),
            "total_rollouts": result.total_rollouts,
            "clustering_method": result.clustering_method,
            "mi_by_position_decile": [
                {"decile": d, "mean_mi": m, "n_boundaries": n}
                for d, m, n in result.mi_by_position_decile
            ],
            "gate": {
                "value": result.mean_mi_bits,
                "pass": result.mean_mi_bits > gate_pass_bits,
                "marginal": gate_nli_bits < result.mean_mi_bits <= gate_pass_bits,
                "fail": result.mean_mi_bits <= gate_nli_bits,
            },
        },
    )
    wandb_run.log_artifact(json_path, artifact_type="concordance", name="concordance_mi")

    # CSV per-decile
    csv_path = out_dir / "concordance_per_position.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        import csv

        w = csv.writer(f)
        w.writerow(["decile", "mean_mi_bits", "n_boundaries"])
        for d, m, n in result.mi_by_position_decile:
            w.writerow([d, f"{m:.6f}", n])
    wandb_run.log_artifact(csv_path, artifact_type="concordance", name="concordance_per_position")

    # Raw per-boundary record (heavy — used for debugging / NLI fallback decision)
    raw_path = out_dir / "concordance_raw.jsonl"
    write_jsonl(
        raw_path,
        (
            {
                "trajectory_id": r.trajectory_id,
                "boundary_idx": r.boundary_idx,
                "token_position": r.token_position,
                "relative_position": r.relative_position,
                "n_samples": r.n_samples,
                "prefix_clusters": r.prefix_clusters,
                "terminal_clusters": r.terminal_clusters,
                "mi_bits": r.mi_bits,
                "lookahead_texts": r.lookahead_texts,
                "terminal_texts": r.terminal_texts,
                "prefix_labels": r.prefix_labels,
                "terminal_labels": r.terminal_labels,
            }
            for r in result.per_boundary
        ),
    )
    wandb_run.log_artifact(raw_path, artifact_type="concordance", name="concordance_raw")

    # wandb metrics
    wandb_run.log(
        {
            "concordance/mean_mi_bits": result.mean_mi_bits,
            "concordance/median_mi_bits": result.median_mi_bits,
            "concordance/n_boundaries": len(result.per_boundary),
            "concordance/total_rollouts": result.total_rollouts,
            "concordance/gate_pass": bool(result.mean_mi_bits > gate_pass_bits),
            "concordance/gate_marginal": bool(
                gate_nli_bits < result.mean_mi_bits <= gate_pass_bits
            ),
            "concordance/gate_fail": bool(result.mean_mi_bits <= gate_nli_bits),
        }
    )
    wandb_run.log_table(
        "concordance/mi_by_position",
        columns=["decile", "mean_mi_bits", "n_boundaries"],
        rows=[[d, m, n] for d, m, n in result.mi_by_position_decile],
    )
    wandb_run.log_summary(
        mean_mi_bits=result.mean_mi_bits,
        median_mi_bits=result.median_mi_bits,
        n_boundaries=len(result.per_boundary),
        gate_decision=(
            "pass"
            if result.mean_mi_bits > gate_pass_bits
            else "marginal" if result.mean_mi_bits > gate_nli_bits else "fail"
        ),
    )

    print("\nConcordance summary")
    print("-" * 40)
    print(f"  mean MI (bits):   {result.mean_mi_bits:.3f}")
    print(f"  median MI (bits): {result.median_mi_bits:.3f}")
    print(f"  n_boundaries:     {len(result.per_boundary)}")
    print(f"  rollouts used:    {result.total_rollouts}")
    gate = (
        "PASS" if result.mean_mi_bits > gate_pass_bits
        else "MARGINAL — run NLI fallback" if result.mean_mi_bits > gate_nli_bits
        else "FAIL — pivot to Plan B"
    )
    print(f"  gate decision:    {gate}")

    wandb_run.finish()


if __name__ == "__main__":
    app()
