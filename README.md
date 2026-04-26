# grpocredit — VoI-allocated credit assignment for GRPO

Implementation of the research proposal *Value-of-Information Allocation for GRPO Credit Assignment* (see `../research_proposal_grpo_clean.md`). Sprint phase (Apr 24 – Apr 27) and offline oracle experiments landed in `scripts/sprint_*`; main training phase (Apr 28 – May 25) lands in `scripts/main_*` once the sprint gate passes.

## What is in here

| Layer | What it does | Files |
|---|---|---|
| `common/` | configs (pydantic + YAML), wandb wrapper, shared types | `config.py` `logging.py` `types.py` `utils.py` |
| `rollout/` | vLLM / HF rollouts with forced-token support, GSM8K / MATH loaders, sympy verifier, syntactic boundary detector | `vllm_runner.py` `hf_runner.py` `boundary.py` `verifier.py` `datasets.py` |
| `voi/` | Stage 0 group filter, Stage 1 token-entropy × w_pos, Stage 2 K_LA=4 lookahead + sentence-T5 clustering, CUSUM, cascade orchestrator | `stage0_group_filter.py` `stage1_entropy.py` `stage2_semantic.py` `cusum_aux.py` `cascade.py` |
| `advantage/` | TD-style segment deltas over probed pivots + James–Stein shrinkage | `segment_gae.py` `shrinkage.py` |
| `oracle/` | Q^π-variance oracle, concordance MI, κ, position curve — all offline | `q_variance_oracle.py` `concordance_check.py` `kappa_estimator.py` `position_curve.py` |

## Quick start

```bash
# 1. Env (do on Linux cluster — vLLM + sentence-transformers)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[vllm,dev]"

# 2. wandb
export WANDB_API_KEY=...       # or `wandb login`
export WANDB_PROJECT=grpo-voi
export WANDB_ENTITY=<your-entity>

# 3. Sprint Day 1 — infra smoke
python scripts/sprint_d1_infra_smoke.py \
    --config configs/base_qwen_math.yaml \
    --n-trajectories 100 \
    --output-dir experiments/sprint

# 4. Sprint Day 2 (parallel; two GPU groups)
python scripts/sprint_d2_concordance.py --config configs/base_qwen_math.yaml   # ~15K rollouts
python scripts/sprint_d2_oracle.py       --config configs/base_qwen_math.yaml   # ~96K rollouts

# 5. Sprint Day 3 — gate report
python scripts/sprint_d3_gate_report.py --sprint-dir experiments/sprint
```

Each script writes JSON/CSV under `experiments/sprint/` *and* logs metrics/artifacts to wandb. Results of the Day 3 gate (`GATE_REPORT.md`) lock the main-experiment plan for the next 4 weeks — see `experiment_plan_grpo_voi.md` §2.

## Server2 runbooks

Three runbooks cover the production launch on server2 (single 80 GB GPU is enough per phase). They differ in scope:

| Runbook | Scope | Status |
|---|---|---|
| `SERVER2_RUNBOOK.md` | Master phase index — A → B1 → B2a → B2b. Detailed for Phase A; for B1 it punts to the VinePPO doc. | A ready; B2a/B2b are placeholders |
| `../VinePPO/SERVER2_RUNBOOK.md` | B1 subsystem reference — VinePPO repro + treetune-internal GRPO. | Ready |
| `SERVER2_RUNBOOK_paired_runs.md` | B1 orchestration — sanity → VinePPO → GRPO → oracle on each GRPO iter. Adds the per-iter oracle wrapper on top of what the other two cover. | Ready |

Execution order on a fresh server2:

1. **One-time setup.** Create both conda envs, download the 4 checkpoints, log into wandb. Use `SERVER2_RUNBOOK.md` §0–§1 for the `grpocredit` env and `../VinePPO/SERVER2_RUNBOOK.md` §1.1–§1.4 for the `vineppo` env.
2. **Pre-flight sanity (every session).** `bash scripts/sanity_check_server2.sh` — fails fast on env / test / cache problems before you burn GPU hours.
3. **Phase A — oracle on the 4 starting policies.** Follow `SERVER2_RUNBOOK.md` §2. Cheapest first (rho-1b, ~30 min), then DeepSeek-MATH-v2 / DeepSeek-GSM8K (~3–4 h each), then Qwen-Instruct as the saturation-ceiling probe. **Gating step** — exit code 6 on the §5 group-variance gate means do not RL on that `π_ref`. Total ~7–9 h.
4. **Phase B1 — paired runs + per-iter oracle.** Follow `SERVER2_RUNBOOK_paired_runs.md` end-to-end (it is the umbrella). Its §2 launches VinePPO, §3 launches matched-trainer GRPO, §4 oracles each emitted GRPO checkpoint. The other two runbooks are subsystem references that §2/§3 of this doc point into for deeper detail. Total ~10–14 h on rho-1b, 4–6× that on 7B.
5. **Phase B2a / B2b** — code not yet written; placeholders in the master runbook.

For B1 work specifically: `SERVER2_RUNBOOK_paired_runs.md` is the one you follow top-to-bottom; the other two are subsystem references it points into. Phase A is the prerequisite that decides which starting policies are worth RL'ing on.

## Committed sprint decisions (do not re-litigate)

| ID | Decision | Source |
|---|---|---|
| D1 | Matched-compute regime at 90 rollouts/trajectory | plan §0 |
| D2 | Binary Stage-2 gate at $K_{\text{LA}}=4$ | plan §0 |
| D3 | Qwen2.5-Math-7B-Instruct primary; DeepSeekMath-7B repro | plan §0 |
| D4 | GSM8K+MATH joint train, AIME24/Olympiad/MATH-500 OOD | plan §0 |
| D5 | verl RL framework | plan §0 |

## Go / no-go gate thresholds

| Check | Pass | Marginal | Fail |
|---|---|---|---|
| Concordance MI | $> 0.3$ bits | $0.15$–$0.3$ (NLI fallback) | $\le 0.15$ → Plan B |
| $\rho(s_2, \widehat{\mathrm{Var}}_{a\sim\pi}(Q^\pi))$ | 95% CI low $\ge \rho_{\text{gate}}$ | straddles | below |
| $\kappa$ | $> 3$ | $2$–$3$ | $< 2$ |

$\rho_{\text{gate}} = \sqrt{0.10 / (f_{\text{sel}} \cdot \kappa)}$, $f_{\text{sel}} = 0.15$.
