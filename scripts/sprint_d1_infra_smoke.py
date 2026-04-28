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
from grpocredit.oracle.group_variance import (  # noqa: E402
    compute_group_variance_report,
    grouped_rewards_from_runner_output,
)
from grpocredit.oracle.rollout_diversity import (  # noqa: E402
    RolloutDiversityError,
    assert_diverse_rollouts,
    diversity_probe,
)
from grpocredit.oracle.stop_gate import classify_stop_gate  # noqa: E402
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
    group_variance_probe_prompts: int = typer.Option(
        256,
        help=(
            "Number of distinct prompts for the §5 group-variance gate. "
            "Set to 0 to skip the probe (e.g., for a fast smoke test)."
        ),
    ),
    group_variance_G: int = typer.Option(
        8, help="Group size G for the §5 group-variance gate (default 8 = main GRPO size)."
    ),
    group_variance_split: str = typer.Option(
        "test",
        help=(
            "Dataset split for the §5 group-variance gate. "
            "Default 'test' because the runbook §2.1 target bands "
            "(rho-1b ≈ 0.60–0.85, DeepSeek-SFT'd ≈ 0.85–0.95, "
            "Qwen-Instruct ≈ 0.30–0.40) were calibrated against "
            "VinePPO's test-split pass@1 numbers. Measuring on 'train' "
            "biases fraction_informative downward for SFT'd models "
            "because they saturate in-sample (rho-1b on GSM8K train "
            "gives pass@1≈0.58 vs 0.35 on test → too many groups "
            "collapse to all-correct). Use 'train' if you specifically "
            "want an 'in-sample RL signal budget' measurement."
        ),
    ),
    stop_gate_min_group_variance_frac: float = typer.Option(
        0.5,
        help=(
            "§5 gate: fraction of G-groups with informative (Var(r) > 0) reward. "
            "Below this, RL on this `π_ref` is a bad bet — see "
            "research_plan/sft_warmup_plan.md §5 for the reasoning."
        ),
    ),
    proceed_on_policy_gate_fail: bool = typer.Option(
        False,
        "--proceed-on-policy-gate-fail",
        help=(
            "If set, a group-variance gate failure alone is NOT treated as a "
            "stop-gate — the script exits 0 after loudly logging the failure "
            "to wandb and disk. Infra failures (boundaries_mean < threshold "
            "or verifier_accuracy < threshold) still exit non-zero because "
            "those are bugs, not policy-quality signals. Use this for "
            "intentionally-weak starting policies (e.g. rho-1b as a "
            "fast-iteration debug policy, per plan §3.A) where you want "
            "Day 2/3 oracle numbers despite a borderline gate."
        ),
    ),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(config, overrides={"output_dir": output_dir, "name": run_name})
    seed_everything(cfg.seed)
    out_dir = ensure_dir(output_dir)

    wandb_run = init_wandb(cfg, run_name=run_name,
                            extra_config={"sprint_day": 1})

    # ── 1. Load a small math dataset for trajectory generation ─────
    traj_dataset = cfg.data.train_datasets[0] if cfg.data.train_datasets else "math"
    log.info("Loading %d %s[train] prompts for trajectory generation",
             n_trajectories, traj_dataset)
    prompts = load_prompts(traj_dataset, split="train", n=n_trajectories)
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

    # ── 1b. Rollout-diversity sentinel ──────────────────────────────
    # Cheap pre-gate: sample G=4 short rollouts on 8 prompts and assert the
    # runner is not silently collapsing to duplicates. See the failure mode
    # documented in `grpocredit.rollout.vllm_runner`'s module docstring and
    # tracked in `oracle.rollout_diversity.assert_diverse_rollouts`. If this
    # trips, the 256×G group-variance gate below would be meaningless — bail
    # with a clear error rather than burn GPU time on a rigged probe.
    log.info("Rollout-diversity sentinel: 8 prompts × G=4")
    probe_src = prompts[: min(8, len(prompts))]
    probe_formatted = build_prompts(probe_src, tokenizer, template=cfg.data.prompt_template)
    try:
        probe_texts = diversity_probe(
            runner,
            probe_prompts=probe_formatted,
            n_per_prompt=4,
            max_new_tokens=64,
            seed=cfg.seed + 13,  # dropped inside the runner for n>1; see docs
        )
        diversity_report = assert_diverse_rollouts(probe_texts)
    except RolloutDiversityError as e:
        log.error("%s", e)
        wandb_run.log_summary(
            fatal_error=str(e),
            rollout_diversity_failed=True,
        )
        wandb_run.finish(exit_code=7)
        raise typer.Exit(7) from e
    log.info(
        "Rollout-diversity sentinel PASS: mean_unique_fraction=%.3f, "
        "all_identical=%d/%d",
        diversity_report.mean_unique_fraction,
        diversity_report.n_groups_all_identical,
        diversity_report.n_groups,
    )
    wandb_run.log(
        {
            f"rollout_diversity/{k}": v
            for k, v in diversity_report.to_dict().items()
        }
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
    # NOTE: intentionally always probes the MATH dataset regardless of
    # the config's `train_datasets`. This is a cross-verifier sanity
    # check — we want to know that `MathVerifier.score(gt_solution,
    # gt_answer)` agrees with itself on MATH's `\boxed{}` convention,
    # which is the hardest extraction path. If this passes, the verifier
    # is correctly wired. A failure here means `math_verify` is broken
    # in the env, not a model/policy issue.
    log.info(
        "Verifier sanity check on %d MATH solutions (ground truth, "
        "cross-dataset — not affected by config.data.train_datasets)",
        verifier_probe_size,
    )
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

    # ── 6. Group-variance gate (sft_warmup_plan.md §5) ────────────
    # Sample N prompts × G rollouts, score, count fraction of groups with
    # Var(reward) > 0. PASS gate: ≥ stop_gate_min_group_variance_frac.
    # On a saturated start (e.g., Qwen-Instruct, pass@1≈0.95) most groups
    # are degenerate and this fails — exactly the case the plan flags as
    # "do NOT start RL here". The metric also doubles as a paper figure
    # ("informative-group fraction at step 0 across starting policies").
    gv_report_dict = None
    if group_variance_probe_prompts > 0:
        gv_dataset = (
            cfg.data.train_datasets[0] if cfg.data.train_datasets else "math"
        )
        log.info(
            "Group-variance gate: %d prompts × G=%d rollouts on %s[%s]",
            group_variance_probe_prompts,
            group_variance_G,
            gv_dataset,
            group_variance_split,
        )
        try:
            gv_prompts = load_prompts(
                gv_dataset,
                split=group_variance_split,
                n=group_variance_probe_prompts,
            )
        except Exception as e:
            log.warning(
                "load_prompts(%s, split=%s) failed (%s); falling back to 'train' "
                "— this is expected for datasets that only ship one split "
                "(e.g. AIME, MATH-500).",
                gv_dataset,
                group_variance_split,
                e,
            )
            gv_prompts = load_prompts(
                gv_dataset, split="train", n=group_variance_probe_prompts
            )
        gv_formatted = build_prompts(
            gv_prompts, tokenizer, template=cfg.data.prompt_template
        )
        gv_groups = runner.generate_from_prompts(
            gv_formatted,
            n_per_prompt=group_variance_G,
            seed=cfg.seed + 7,
        )
        gv_rewards = grouped_rewards_from_runner_output(
            gv_groups,
            verifier,
            ground_truth_answers=[p.ground_truth_answer for p in gv_prompts],
        )
        gv_report = compute_group_variance_report(gv_rewards)
        gv_report_dict = gv_report.to_dict()
        gv_report_dict["G"] = group_variance_G
        gv_report_dict["probe_prompts"] = group_variance_probe_prompts
        gv_report_dict["pass_threshold"] = stop_gate_min_group_variance_frac
        gv_report_dict["pass"] = (
            gv_report.fraction_informative >= stop_gate_min_group_variance_frac
        )
        write_json(out_dir / "day1_group_variance.json", gv_report_dict)
        wandb_run.log_artifact(
            out_dir / "day1_group_variance.json",
            artifact_type="group_variance",
            name="day1_group_variance",
        )
        wandb_run.log(
            {
                "group_variance/fraction_informative": gv_report.fraction_informative,
                "group_variance/n_groups": gv_report.n_groups,
                "group_variance/n_informative": gv_report.n_informative,
                "group_variance/n_all_correct": gv_report.n_groups_all_correct,
                "group_variance/n_all_wrong": gv_report.n_groups_all_wrong,
                "group_variance/mean_group_reward_mean": gv_report.mean_group_reward_mean,
                "group_variance/mean_group_reward_std": gv_report.mean_group_reward_std,
                "group_variance/G": group_variance_G,
                "group_variance/pass": gv_report_dict["pass"],
            }
        )
        log.info(
            "Group-variance gate: %.3f informative-group fraction (%d/%d) — %s",
            gv_report.fraction_informative,
            gv_report.n_informative,
            gv_report.n_groups,
            "PASS" if gv_report_dict["pass"] else "FAIL",
        )

    # ── Stop-gate evaluation ───────────────────────────────────────
    # Infra vs policy classification lives in `grpocredit.oracle.stop_gate`
    # (see its module docstring for the full rationale). Summary:
    #   exit 1  = infra  (detector_max==0 or verifier_acc<min) — unwaivable
    #   exit 6  = policy (boundaries_mean<min or §5 gate fail) — waivable
    #   exit 0  = pass, or policy-only fail with the override set
    decision = classify_stop_gate(
        boundaries_mean=mean_b,
        boundaries_max=max_b,
        verifier_accuracy=verifier_acc,
        gv_pass=(gv_report_dict["pass"] if gv_report_dict is not None else None),
        min_boundaries=stop_gate_min_boundaries,
        min_verifier_accuracy=stop_gate_min_verifier_acc,
        proceed_on_policy_gate_fail=proceed_on_policy_gate_fail,
    )

    gate_report = {
        "n_trajectories": n_trajectories_actual,
        "boundaries_mean": mean_b,
        "boundaries_min": min_b,
        "boundaries_max": max_b,
        "verifier_accuracy": verifier_acc,
        "group_variance": gv_report_dict,
        "infra_fail": decision.infra_fail,
        "policy_fail": decision.policy_fail,
        "proceed_on_policy_gate_fail": decision.proceed_on_policy_gate_fail,
        "stop_gate_reasons": list(decision.reasons),
        "stop_gate_triggered": decision.effective_stop,
        "pass": not decision.effective_stop,
    }
    write_json(out_dir / "day1_summary.json", gate_report)
    wandb_run.log_summary(**gate_report)

    print("\nDay 1 smoke-test summary")
    print("-" * 40)
    for k, v in gate_report.items():
        print(f"  {k}: {v}")
    print("-" * 40)
    if decision.reasons:
        print("Gate-check triggers:")
        for r in decision.reasons:
            print(f"  - {r}")
    if decision.effective_stop:
        if decision.infra_fail:
            print(
                "STOP-GATE TRIGGERED (INFRA) — detector or verifier is broken. "
                "Fix the code/config before Day 2. --proceed-on-policy-gate-fail "
                "does NOT waive this."
            )
        else:
            print(
                "STOP-GATE TRIGGERED (POLICY) — one of {short CoTs, low §5 "
                "fraction_informative}. These are distribution properties of "
                "π_ref, not code bugs. Re-run with --proceed-on-policy-gate-fail "
                "to collect Day 2/3 oracle numbers anyway (recommended for "
                "rho-1b and other intentionally-weak debug policies)."
            )
    elif decision.policy_fail and decision.proceed_on_policy_gate_fail:
        print(
            "POLICY GATE FAILED but --proceed-on-policy-gate-fail was set; "
            "continuing to Day 2. The failure is recorded in day1_summary.json "
            "and wandb summary; Day 3 GATE_REPORT.md will still flag it."
        )
    else:
        print("PASS")

    # Exit codes match SERVER2_RUNBOOK.md §2 table:
    #   0 = all pass (or policy-fail overridden)
    #   1 = infra fail (detector broken or verifier broken)
    #   6 = policy fail (short CoTs or group-variance gate; overrideable)
    wandb_run.finish(exit_code=decision.exit_code)
    if decision.exit_code != 0:
        raise typer.Exit(decision.exit_code)


if __name__ == "__main__":
    app()
