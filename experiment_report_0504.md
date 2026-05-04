# Experiment Report — 2026-05-04

End-to-end sprint run on **rho-1b-sft-GSM8K** with the new selection-metric
pipeline (per-trajectory top-1, overlap@2, κ_emb / κ_sig added alongside
Spearman). All four phases ran clean; the gate verdict is
`wrong_starting_policy` (`exit_code: 6`), as expected for a debug-tier policy.

wandb runs:
- d1 infra-smoke: <https://wandb.ai/suesie/grpo-voi/runs/17g4dc9i>
- d2 concordance: <https://wandb.ai/suesie/grpo-voi/runs/gsmhdtuq>
- d2 oracle:      <https://wandb.ai/suesie/grpo-voi/runs/sebpde22>
- d3 gate:        <https://wandb.ai/suesie/grpo-voi/runs/5hseq10h>

---

## Day 2A — embedding-variance diagnostic

```
informative prompts:   124/256
boundaries analysed:   229
mean remaining tokens: 77.5
```

| | Config A (K=4 / len=30) | Config B (K=8 / len=15 / rew=4) |
|---|---|---|
| **top-1 agreement** | **0.561** (66 trajs ≥2 b.) | 0.485 (66 trajs ≥2 b.) |
| **overlap@2** | **0.590** (13 trajs ≥4 b.) | 0.410 (13 trajs ≥4 b.) |
| **κ_emb** | **1.651** (49 trajs, reward_var > 0) | 1.363 (42 trajs, reward_var > 0) |
| Spearman ρ (legacy) | 0.298 [0.175, 0.412] | 0.160 [0.031, 0.283] |
| Spearman long-only | 0.144 (n=88, remaining > 60) | 0.089 (n=187, remaining > 30) |

**Verdict.** Config A clears the per-trajectory gate (κ_emb > 1.5, top-1 > 0.50). Config B's 15-token lookahead (~19 % of mean remaining 78) is too short — adding rollouts at K=8 doesn't pay for the lost horizon.

---

## Day 2B — Q^π-variance oracle (with selection metrics)

```
informative prompts: 126/256
oracle records:      288
total rollouts:      55,296
h_fwd_k:             10
```

### Per-trajectory selection metrics (primary)

| Signal | top-1 | κ_sig | overlap@2 | n_traj |
|---|---|---|---|---|
| **H_token** | **0.584** | **1.609** | **0.573** | 77 |
| s2 | 0.558 | 1.494 | 0.573 | 77 |
| H_fwd | 0.506 | 1.289 | 0.360 | 77 |

### Spearman ρ vs Var(Q^π) (legacy global)

| Signal | ρ | 95 % CI |
|---|---|---|
| H_token | 0.191 | [0.074, 0.304] |
| H_fwd | 0.155 | [0.036, 0.269] |
| H_sem | **−0.128** | [−0.243, −0.008] |
| s2 | NaN | [NaN, NaN] |

```
κ            = 1.762   CI [0.655, 2.986]   selection_score = h_fwd
ρ_gate       = 0.615
position_shape = bimodal
group_variance: 126/256 informative (fraction = 0.492)
```

**Findings.**

1. **H_token is the best selector.** Single-token entropy at the boundary beats `H_fwd` (10-token forward average) on all three per-trajectory metrics (0.584 vs 0.506 top-1, 1.609 vs 1.289 κ_sig, 0.573 vs 0.360 overlap@2).
2. **Why H_fwd loses on rho-1b.** Mean ~2.3 boundaries/traj, ~50-token CoTs. K=10 is ~20 % of the response and dilutes the load-bearing single-token decision (e.g. multiply-vs-divide) by averaging in 9 deterministic execution tokens after the choice.
3. **H_sem is *anti*-correlated** (ρ = −0.128). Boundaries where all 4 lookaheads cluster together (`H_sem = 0`) actually have higher Var(Q^π) than multi-cluster boundaries — the binary cosine gate is worse than useless on this policy.
4. **Spearman undersells the cheap signals.** H_token's ρ = 0.191 looks weak, but per-traj κ_sig = 1.609 means probing its argmax boundary captures 61 % more outcome variance than random. Confirms that ρ pools the wrong axes.

---

## Day 3 — gate report

| Gate | Value | Status | Comment |
|---|---|---|---|
| Group-variance | 0.492 | **FAIL** | Expected for the rho-1b debug policy |
| Concordance | top-1 = 0.561, κ_emb = 1.651 | **PASS** | Embedding lookahead works as a per-traj selector |
| κ | 1.76 [0.66, 2.99] | **FAIL** | Below 3.0 (improved from 1.25 → 1.76 after switching to informative groups) |
| ρ(s₂, Var(Q^π)) | NaN | **FAIL** | s₂ collapsed — H_sem = 0 nearly everywhere |

```
overall:    wrong_starting_policy
exit_code:  6
next_steps: Group-variance gate FAILED at step 0 — starting policy is too
            saturated or too weak (sft_warmup_plan.md §5). DO NOT proceed to
            RL. Switch π_ref to a SFT-warmed model
            (Option B: realtreetune/deepseekmath-7b-sft-{MATH-v2,GSM8K};
             Option A: SFT Qwen2.5-Math-7B base ourselves).
            Re-run sprint Day 1+.
```

κ has now been 1.25 → 2.16 → 1.76 across re-runs; CI [0.66, 2.99] shows the true value is genuinely uncertain at n ≈ 288 records. `position_shape` has likewise oscillated `mid_peak → flat → bimodal` — both confirm the sample is too thin to classify on rho-1b.

---

## Implications for the method

1. **Stage 2 (cosine clustering) is broken on short-CoT models.** H_sem is anti-correlated with Var(Q^π); the binary gate filters out the best boundaries. On rho-1b the effective cascade should be **Stage 0 → Stage 1 (H_token × w_pos) → probe**, no Stage 2.
2. **H_token alone is a viable VoI selector.** κ_sig = 1.609 ≈ "probe the highest-entropy boundary, get 61 % more outcome variance than random." This is the `voi_stage1` config — keep it.
3. **H_fwd needs auto-scaling for short CoTs.** Two options: (a) `K = min(10, remaining_tokens // 5)`, or (b) treat H_fwd as a long-CoT-only signal and let DeepSeekMath be its real test.
4. **The exit-6 verdict is correct, and the infrastructure is validated.** All signals fire, all artefacts log; the policy itself just doesn't leave room for VoI to matter. DeepSeekMath-SFT (8–15 boundaries/traj, 200+ token CoTs) is the headline run.

---

## Appendix: full pipeline trace for `PROCEED_ON_POLICY_GATE_FAIL=1 bash scripts/run_oracle.sh configs/oracle/rho1b_sft_gsm8k.yaml`

### 0. Shell setup (`run_oracle.sh`)

1. `set -euo pipefail` — any non-zero exit kills the pipeline.
2. `CONFIG=configs/oracle/rho1b_sft_gsm8k.yaml`, `N_TRAJ=100`, `PROBE=200`.
3. Because `PROCEED_ON_POLICY_GATE_FAIL=1`, sets `DAY1_EXTRA_ARGS=(--proceed-on-policy-gate-fail)` and prints an advisory message.
4. Resolves `output_dir` by importing `load_config` inline — the YAML has `output_dir: experiments/oracle/rho1b_sft_gsm8k`. The `extends: ../base_qwen_math.yaml` merge pulls in all base defaults, then the rho1b YAML overrides `model`, `data`, `rollout`, `wandb`, `name`, and `output_dir`.
5. `mkdir -p experiments/oracle/rho1b_sft_gsm8k`, appends a timestamped provenance line to `launch.log`.

**Effective merged config highlights:**
- Model: `realtreetune/rho-1b-sft-GSM8K` (1.4B params, bfloat16)
- Data: `train_datasets: [gsm8k]`, template: `vineppo_math_task`
- Rollout: `temperature=0.35`, `top_p=0.9`, `stop=["\n\n\nProblem:"]`, `max_new_tokens=1024`
- Oracle: `boundaries_per_trajectory=5`, `top_m_actions=6`, `rollouts_per_forced_action=32`, `h_fwd_k=10`
- Output: `experiments/oracle/rho1b_sft_gsm8k/`

### 1. Day 1 — `sprint_d1_infra_smoke.py --proceed-on-policy-gate-fail`

Called with `--config configs/oracle/rho1b_sft_gsm8k.yaml --n-trajectories 100 --verifier-probe-size 200 --output-dir experiments/oracle/rho1b_sft_gsm8k --proceed-on-policy-gate-fail`.

**Step 1a: Rollout-diversity sentinel.** 8 prompts × G=4 short (64-token) rollouts. `assert_diverse_rollouts()` checks rollouts aren't collapsing to duplicates. Exit 7 if broken (unwaivable).

**Step 1b: Generate 100 trajectories.** Loads 100 GSM8K[train] prompts. Loads vLLM with `realtreetune/rho-1b-sft-GSM8K`. Formats with `vineppo_math_task` template (`"[MATH_TASK] Problem:\n{q}\n\nSolution:"`). Generates 1 rollout per prompt (n_per_prompt=1), temp=0.35, stop on `"\n\n\nProblem:"`. Scores with `MathVerifier`, writes `day1_rollouts.jsonl`.

**Step 2: Boundary detection.** Detects sentence/step boundaries in each trajectory (max 30/traj, min 8 tokens between). Writes `day1_boundaries.json`. For rho-1b on GSM8K: CoTs are short (2-3 steps), so `boundaries_mean` is ~2-4, below the default `stop_gate_min_boundaries=3.0` — a **policy fail**, not infra.

**Step 3: Verifier sanity check.** 200 **MATH** (not GSM8K!) problems, checks the verifier against its own ground truth. Intentionally cross-dataset (tests `\boxed{}` extraction). Should pass easily (≥0.95).

**Step 4: Stage-2 clustering smoke test.** 5 boundaries, 4 short lookahead continuations each, sentence-T5 clustering. Writes `day1_clustering.json`.

**Step 5: Group-variance gate (§5).** 256 GSM8K[**test**] prompts × G=8 = 2048 rollouts. Computes `fraction_informative`. rho-1b on GSM8K-test has pass@1 ≈ 0.35, expected fraction ~0.60-0.85. Writes `day1_group_variance.json`.

**Step 6: Stop-gate classification.** `classify_stop_gate()` separates infra vs policy:
- **Infra** (unwaivable, exit 1): `boundaries_max == 0` OR `verifier_accuracy < 0.9`.
- **Policy** (waivable, exit 6 → 0 with override): `boundaries_mean < 3.0` OR `fraction_informative < 0.5`.
- With `--proceed-on-policy-gate-fail`: policy failures are logged but `effective_stop = False`, `exit_code = 0`. Pipeline continues.

**Day 1 outputs:**

| File | Contents |
|---|---|
| `day1_rollouts.jsonl` | 100 trajectories (prompt, response, reward, token_ids) |
| `day1_boundaries.json` | Boundary maps + stats |
| `day1_verifier_accuracy.txt` | Verifier self-check accuracy |
| `day1_clustering.json` | Stage-2 smoke test results |
| `day1_group_variance.json` | §5 gate: fraction_informative, pass/fail, breakdown |
| `day1_summary.json` | Combined gate report (infra_fail, policy_fail, reasons, pass flag) |

### 2. Day 2A — `sprint_d2_concordance.py`

Called with `--config configs/oracle/rho1b_sft_gsm8k.yaml --output-dir experiments/oracle/rho1b_sft_gsm8k`.

**Step 1: Group-variance probe (independent re-run).** 256 GSM8K[**test**] × G=8 = 2048 rollouts. Identifies informative prompts.

**Step 2: Pick one trajectory per informative prompt.** First rollout from each informative group.

**Step 3: Boundary detection + sampling.** Up to 8 per trajectory, evenly spaced.

**Step 4: Embedding-variance diagnostic — two configs.**

*Config A (K=4, len=30):* For each sampled boundary: 4 continuations of 30 tokens each. Encodes with sentence-T5, computes embedding variance (cosine distance). Also 4 terminal (up to 512 tokens) continuations → reward variance. Correlates emb_var vs reward_var.

*Config B (K=8, len=15, reward from 4):* 8 short (15-token) lookaheads for embedding, 4 full-length for reward. Tests cheaper/more numerous previews.

**Metrics:** per-trajectory top-1 agreement, κ_emb, overlap@2, global Spearman ρ(emb_var, reward_var) with Fisher-z CI.

**Day 2A outputs:**

| File | Contents |
|---|---|
| `emb_var_summary.json` | Config A + B selection metrics, Spearman ρ, group_variance |
| `emb_var_per_boundary.jsonl` | Per-boundary emb_var, reward_var, position, etc. |
| `emb_var_per_position.csv` | Decile breakdown of emb/reward variance |
| `boundary_annotated_responses.txt` | Human-readable annotated trajectories |

### 3. Day 2B — `sprint_d2_oracle.py`

Called with `--config configs/oracle/rho1b_sft_gsm8k.yaml --output-dir experiments/oracle/rho1b_sft_gsm8k`.

**Step 1: Group-variance probe (third time).** 256 GSM8K[**train**] × G=8 = 2048 rollouts. Filters to informative prompts.

**Step 2: Boundaries + Stage 1 scores.** One trajectory per informative prompt. 5 boundaries per trajectory (from `oracle.boundaries_per_trajectory=5`). Stage1Scorer computes `H_token` and `H_fwd` (K=10) at each boundary.

**Step 3: Q^π-variance oracle (dominant cost).** For each sampled boundary: top M=6 most-likely next tokens (forced actions) × K=32 full continuations each → per-action Q estimate → `Var(Q^π)`. Tail stratum (16 rollouts) if coverage < 0.9. Total budget: ~(n_boundaries × 6 × 32) + tail ≈ **~96k rollouts** (feasible on 1.4B).

**Step 4: Correlations.** Spearman ρ(H_token, Var(Q^π)) and ρ(H_fwd, Var(Q^π)) with Fisher-z CIs.

**Step 4b: Per-trajectory selection metrics.** For H_fwd and H_token: top-1 agreement, κ_signal, overlap@2.

**Step 5: κ estimation.** Selects best signal (H_fwd if it has variance, else H_token). κ = Var(Q^π) at VoI-selected / Var(Q^π) at random, with bootstrap CI. `rho_gate = sqrt(f_target / (f_sel · κ))`.

**Step 6: Position curve.** Decile breakdown of Var(Q^π) by relative position → shape classification (flat/tent/bimodal). Writes `position_lookup.csv` for Stage-1 `w_pos='lookup'`.

**Day 2B outputs:**

| File | Contents |
|---|---|
| `oracle_q_variance.json` | Per-boundary Var(Q^π), H_token, H_fwd, coverage, tail results |
| `oracle_correlations.json` | ρ(H_token, Var(Q^π)), ρ(H_fwd, Var(Q^π)) + CIs |
| `oracle_selection_metrics.json` | Per-trajectory top-1, κ_signal, overlap@2 for each signal |
| `oracle_kappa.txt` | κ, CI, ρ_gate, selection_score, h_fwd_k |
| `oracle_summary.json` | Everything consolidated |
| `oracle_position_curve.csv` | Decile curve |
| `position_lookup.csv` | Normalized curve for Stage-1 w_pos |

### 4. Day 3 — `sprint_d3_gate_report.py`

Called with `--sprint-dir experiments/oracle/rho1b_sft_gsm8k --config configs/oracle/rho1b_sft_gsm8k.yaml`.

Reads all Day 1/2 outputs and evaluates four gates:

| Gate | How evaluated | Likely rho-1b outcome |
|---|---|---|
| **Group variance** (§5) | `fraction_informative ≥ 0.5` from `day1_group_variance.json` | Likely **pass** (~0.60-0.85 expected) |
| **Concordance** | `top1_agreement ≥ 0.5` AND `kappa_emb ≥ 1.5` from `emb_var_summary.json` | Depends on emb_var vs reward_var on short CoTs |
| **κ** | `κ ≥ 3.0` from `oracle_summary.json` | Uncertain — fewer/shorter boundaries → noisier |
| **Oracle signal** | `max(sel_H_fwd_kappa, sel_H_token_kappa) ≥ 1.5` from `oracle_summary.json` | Uncertain |

**Decision tree:** gv fail → exit 6 · concordance fail → exit 4 (Plan B) · other hard fail → exit 4 · 2+ marginal → exit 3 (pilot) · 1 marginal → exit 2 (caveats) · all pass → exit 0.

**Day 3 outputs:**

| File | Contents |
|---|---|
| `GATE_REPORT.md` | Human-readable decision table |
| `gate_decision.json` | Machine-readable: all metrics + verdicts + overall + exit code |

### Key interaction of `PROCEED_ON_POLICY_GATE_FAIL=1`

This flag **only affects Day 1**. It converts a policy-class failure (short CoTs or low group-variance fraction) from a hard stop (exit 6) into exit 0, allowing Days 2A/2B/3 to run. Infra failures (exit 1) are never waived. The failure is still recorded in `day1_summary.json` (`policy_fail: true`), wandb, and Day 3's `GATE_REPORT.md` evaluates the group-variance gate independently. The flag does not suppress the signal — it lets the expensive oracle pipeline run so you get diagnostic numbers from a known-weak policy.

### Complete file manifest under `experiments/oracle/rho1b_sft_gsm8k/`

```
launch.log
day1_rollouts.jsonl
day1_boundaries.json
day1_verifier_accuracy.txt
day1_clustering.json
day1_group_variance.json
day1_summary.json
boundary_annotated_responses.txt
emb_var_summary.json
emb_var_per_boundary.jsonl
emb_var_per_position.csv
oracle_q_variance.json
oracle_correlations.json
oracle_selection_metrics.json
oracle_kappa.txt
oracle_summary.json
oracle_position_curve.csv
position_lookup.csv
GATE_REPORT.md
gate_decision.json
```

Plus **4 wandb runs** (d1, d2A, d2B, d3) in project `grpo-voi` with tags `[phase_a, oracle, rho1b_sft_gsm8k]`, each with their own artifacts and metrics.
