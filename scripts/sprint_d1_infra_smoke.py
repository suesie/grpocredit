"""Sprint Day 1 — infrastructure smoke test.

Exit criterion (per experiment_plan §2 Day 1):
    experiments/sprint/day1_rollouts.jsonl         (100 trajectories)
    experiments/sprint/day1_boundaries.json        (5–15 boundaries/traj)
    experiments/sprint/day1_verifier_accuracy.txt  (≥ 0.95 on 200-prob check)

Stop-gate: if boundaries/traj < 3 or verifier accuracy < 0.9, bail.

All metrics logged to wandb (project: grpo-voi, job_type: sprint-d1).

Usage
-----
    python scripts/sprint_d1_infra_smoke.py \\
        --config configs/base_qwen_math.yaml \\
        --n-trajectories 100 \\
        --verifier-probe-size 200 \\
        --output-dir experiments/sprint
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
    mean_std,
    seed_everything,
    write_json,
    write_jsonl,
)
from grpocredit.rollout.datasets import load_prompts  # noqa: E402
from grpocredit.rollout.verifier import MathVerifier  # noqa: E402
from grpocredit.voi.stage2_semantic import Stage2Scorer  # noqa: E402

log = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def main(
    config: str = typer.Option(..., help="Path to base YAML config"),
    n_trajectories: int = typer.Option(100, help="Base trajectories to generate"),
    n_per_prompt: int = typer.Option(1, help="Rollouts per prompt (Day 1 uses 1)"),
    verifier_probe_size: int = typer.Option(
        200, help="MATH problems to test the verifier on against ground-truth solutions"
    ),
    stage2_cluster_probe_size: int = typer.Option(
        5, help="Number of boundaries to smoke-test Stage-2 clustering on"
    ),
    output_dir: str = typer.Option("experiments/sprint", help="Where to write outputs"),
    run_name: str = typer.Option("sprint-d1-infra-smoke", help="wandb run name"),
    stop_gate_min_boundaries: float = typer.Option(3.0),
    stop_gate_min_verifier_acc: float = typer.Option(0.9),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(config, overrides={"output_dir": output_dir, "name": run_name})
    seed_everything(cfg.seed)
    out_dir = ensure_dir(output_dir)

    wandb_run = init_wandb(cfg, run_name=run_name,
                            extra_config={"sprint_day": 1})

    # ── 1. Load a small math dataset for trajectory generation ─────
    log.info("Loading %d MATH train prompts", n_trajectories)
    prompts = load_prompts(
        cfg.data.train_datasets[0] if cfg.data.train_datasets else "math",
        split="train",
        n=n_trajectories,
    )
    log.info("Loaded %d prompts", len(prompts))

    # Lazy-import vLLM — environment check
    try:
        from grpocredit.rollout.vllm_runner import VLLMRolloutRunner
    except ImportError as e:
        log.error("vLLM import failed: %s", e)
        wandb_run.log_summary(fatal_error=str(e))
        wandb_run.finish(exit_code=1)
        raise

    runner = VLLMRolloutRunner(cfg.model, cfg.rollout)
    tokenizer = runner.tokenizer

    from scripts._shared import (
        build_prompts,
        detect_all_boundaries,
        results_to_trajectories,
    )

    # ── 2. Generate trajectories ───────────────────────────────────
    log.info(
        "Generating %d trajectories (n_per_prompt=%d, max_new_tokens=%d)",
        n_trajectories,
        n_per_prompt,
        cfg.rollout.max_new_tokens,
    )
    formatted = build_prompts(prompts, tokenizer, template=cfg.data.prompt_template)
    groups = runner.generate_from_prompts(
        formatted,
        n_per_prompt=n_per_prompt,
        seed=cfg.seed,
    )
    verifier = MathVerifier()
    trajectories = results_to_trajectories(prompts, groups, verifier)
    rollouts_path = out_dir / "day1_rollouts.jsonl"
    write_jsonl(rollouts_path, (t.to_dict() for t in trajectories))
    wandb_run.log_artifact(rollouts_path, artifact_type="rollouts", name="day1_rollouts")

    n_trajectories_actual = len(trajectories)
    mean_len, std_len = mean_std([t.length for t in trajectories])
    wandb_run.log({"n_trajectories": n_trajectories_actual,
                   "response_length_mean": mean_len,
                   "response_length_std": std_len})

    # ── 3. Boundary detector histogram ─────────────────────────────
    log.info("Detecting boundaries")
    boundaries_by_traj = detect_all_boundaries(trajectories, tokenizer, cfg.boundary)
    counts = [len(v) for v in boundaries_by_traj.values()]
    mean_b, std_b = mean_std(counts)
    min_b = min(counts) if counts else 0
    max_b = max(counts) if counts else 0

    boundaries_json_path = out_dir / "day1_boundaries.json"
    write_json(
        boundaries_json_path,
        {
            "mean": mean_b,
            "std": std_b,
            "min": min_b,
            "max": max_b,
            "n_trajectories": len(counts),
            "boundaries_per_trajectory": {
                tid: [b.to_dict() for b in bds]
                for tid, bds in boundaries_by_traj.items()
            },
        },
    )
    wandb_run.log_artifact(boundaries_json_path, artifact_type="boundaries", name="day1_boundaries")
    wandb_run.log({
        "boundaries_mean": mean_b,
        "boundaries_std": std_b,
        "boundaries_min": min_b,
        "boundaries_max": max_b,
    })

    try:
        import wandb as _wandb

        if wandb_run.enabled:
            hist = _wandb.Histogram(counts)
            wandb_run.log({"boundaries_histogram": hist})
    except Exception:
        pass

    # ── 4. Verifier sanity check (200 MATH problems) ───────────────
    log.info("Verifier sanity check on %d MATH solutions (ground truth)", verifier_probe_size)
    probe = load_prompts("math", split="train", n=verifier_probe_size)
    correct = 0
    total = 0
    for r in probe:
        if not r.ground_truth_answer:
            continue  # skip unparseable ground truth
        # Use the raw solution string as the "response" — verifier should extract
        # and agree with itself on ground truth.
        v = verifier.score(r.ground_truth_raw, r.ground_truth_answer)
        total += 1
        if v.correct:
            correct += 1
    verifier_acc = correct / total if total > 0 else 0.0
    (out_dir / "day1_verifier_accuracy.txt").write_text(
        f"correct={correct}\ntotal={total}\naccuracy={verifier_acc:.4f}\n",
        encoding="utf-8",
    )
    wandb_run.log({"verifier_accuracy": verifier_acc, "verifier_n": total})

    # ── 5. Stage-2 clustering smoke test ───────────────────────────
    log.info("Stage-2 clustering smoke test")
    s2_scorer = Stage2Scorer(cfg.cascade.stage2)
    # Pick a few trajectories with at least one mid-traj boundary
    cluster_records = []
    pick = [
        (t, boundaries_by_traj[t.trajectory_id][len(boundaries_by_traj[t.trajectory_id]) // 2])
        for t in trajectories
        if boundaries_by_traj.get(t.trajectory_id)
    ][:stage2_cluster_probe_size]
    if pick:
        prefixes = [t.prompt_token_ids + t.token_ids[: b.token_position] for t, b in pick]
        lookaheads = runner.continue_from_prefixes(
            prefix_token_ids=prefixes,
            n_continuations=cfg.cascade.stage2.n_lookaheads,
            max_new_tokens=cfg.cascade.stage2.lookahead_max_new_tokens,
            temperature=cfg.cascade.stage2.lookahead_temperature,
            top_p=0.95,
            seed=cfg.seed + 1,
        )
        for (t, b), rs in zip(pick, lookaheads, strict=False):
            texts = [r.response_text for r in rs]
            _labels, sizes = s2_scorer.cluster_lookaheads(texts)
            cluster_records.append(
                {
                    "trajectory_id": t.trajectory_id,
                    "boundary_idx": b.boundary_idx,
                    "token_position": b.token_position,
                    "n_clusters": len(sizes),
                    "cluster_sizes": sizes,
                    "lookahead_texts": texts,
                }
            )
        write_json(out_dir / "day1_clustering.json", cluster_records)
        n_multi = sum(1 for r in cluster_records if r["n_clusters"] > 1)
        wandb_run.log(
            {
                "stage2_smoke_n_boundaries": len(cluster_records),
                "stage2_smoke_n_multi_cluster": n_multi,
                "stage2_smoke_frac_multi_cluster": (
                    n_multi / len(cluster_records) if cluster_records else 0.0
                ),
            }
        )

    # ── Stop-gate evaluation ───────────────────────────────────────
    stop_gate_triggered = (
        mean_b < stop_gate_min_boundaries or verifier_acc < stop_gate_min_verifier_acc
    )
    gate_report = {
        "n_trajectories": n_trajectories_actual,
        "boundaries_mean": mean_b,
        "boundaries_min": min_b,
        "boundaries_max": max_b,
        "verifier_accuracy": verifier_acc,
        "stop_gate_triggered": stop_gate_triggered,
        "pass": not stop_gate_triggered,
    }
    write_json(out_dir / "day1_summary.json", gate_report)
    wandb_run.log_summary(**gate_report)

    print("\nDay 1 smoke-test summary")
    print("-" * 40)
    for k, v in gate_report.items():
        print(f"  {k}: {v}")
    print("-" * 40)
    print("PASS" if not stop_gate_triggered else "STOP-GATE TRIGGERED — fix infra before Day 2")

    wandb_run.finish(exit_code=1 if stop_gate_triggered else 0)
    if stop_gate_triggered:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
