# grpocredit — VoI-allocated credit assignment for GRPO

Implementation of the research proposal *Value-of-Information Allocation for GRPO Credit Assignment* (see `../research_proposal_grpo_clean.md`). Sprint phase (Apr 24 – Apr 27) and offline oracle experiments landed in `scripts/sprint_*`; main training phase (Apr 28 – May 25) lands in `scripts/main_*` once the sprint gate passes.

## What is in here

| Layer | What it does | Files |
|---|---|---|
| `common/` | configs (pydantic + YAML), wandb wrapper, shared types | `config.py` `logging.py` `types.py` `utils.py` |
| `rollout/` | vLLM / HF rollouts with forced-token support, GSM8K / MATH loaders, sympy verifier, syntactic boundary detector | `vllm_runner.py` `hf_runner.py` `boundary.py` `verifier.py` `datasets.py` |
| `voi/` | Stage 0 group filter, Stage 1 token-entropy × w_pos + H_fwd (multi-step entropy), Stage 2 K_LA=4 lookahead + sentence-T5 clustering, CUSUM, cascade orchestrator | `stage0_group_filter.py` `stage1_entropy.py` `stage2_semantic.py` `cusum_aux.py` `cascade.py` |
| `advantage/` | TD-style segment deltas over probed pivots + James–Stein shrinkage | `segment_gae.py` `shrinkage.py` |
| `oracle/` | Q^π-variance oracle, embedding-variance diagnostic, κ, H_fwd (avg entropy over next K tokens), position curve — all offline | `q_variance_oracle.py` `concordance_check.py` `kappa_estimator.py` `position_curve.py` |

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
python scripts/sprint_d2_oracle.py       --config configs/base_qwen_math.yaml   # informative groups + H_fwd

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

## Config knobs worth knowing about

### Verifier — check the extraction method before trusting the gate

The verifier (`src/grpocredit/rollout/verifier.py`) uses a **priority-ordered
registry** of answer extractors. Each entry is `(method_tag, extractor_fn)`
in `_EXTRACTORS`; `extract_final_answer` runs them in order and returns the
first non-empty result. Current registry (pinned by
`tests/test_verifier.py::test_extractor_registry_order`):

| Priority | Method tag | Convention | Used by |
|---|---|---|---|
| 1 | `gsm8k_hash` | `#### X` at end | GSM8K-SFT'd models (rho-1b, deepseekmath-sft-GSM8K, VinePPO's) |
| 2 | `answer_tag` | `<answer>X</answer>` | DeepSeek-R1 and R1-distilled models |
| 3 | `boxed` | last `\boxed{X}` | Qwen-Math-Instruct, deepseekmath-sft-MATH, any MATH-trained model |
| 4 | `answer_is` | "(the )?(final )?answer (is\|:\|=) X" prose | most instruct-tuned models |
| 5 | `fallback` | last numeric token | weak last-resort; method tag flags untrustworthy |

> ⚠ **Extending the verifier is a first-class requirement when bringing on
> a new model or dataset**, not a polish task. The registry today covers
> GSM8K / MATH / AIME-24 / OlympiadBench / MATH-500 on rho-1b, DeepSeekMath-
> SFT'd, Qwen-Math-Instruct, and R1-family outputs. Any model whose output
> convention is not one of the five above will silently grade through the
> `fallback` path and the §5 gate will report garbage `fraction_informative`
> while looking superficially green. The fix is a 5-line addition to
> `_EXTRACTORS` + a verbatim regression test + a runbook-table row —
> workflow in `SERVER2_RUNBOOK.md` §2.4. Always run
> `scripts/inspect_day1_rollouts.py --tokenizer <repo>` after the first
> Day-1 smoke on any new model; if its `[aggregate] verifier extract
> method:` line is dominated by `fallback`, stop and add an extractor
> before anything else.

> ⚠ **Multiple-choice, code-gen, and proof benchmarks do NOT fit this
> registry** — they need a separate verifier class with the same
> `score(response, ground_truth) -> VerifierResult` contract. The oracle
> pipeline is verifier-agnostic, so swapping is a config knob, not a
> refactor. See `SERVER2_RUNBOOK.md` §2.4 for the decision matrix.

How this bit us in the sprint (so the mechanism is concrete): rho-1b-sft-GSM8K
was emitting `#### 72` correctly on 100/100 rollouts, but the pre-fix
`_ANSWER_IS_RE` also matched bare `=\s*X` in every intermediate CoT step
and was ordered *before* `####` extraction. The verifier returned
"24 clips in May" from `= 24 clips in May` on step 1 and scored the
rollout wrong. Pass@1 was reported as 15 % when it was actually ~70 %.
Caught via `scripts/inspect_day1_rollouts.py`, fixed by reordering
priority and narrowing the `answer_is` regex. The priority order and the
verbatim failing rollouts are now regression-tested in
`tests/test_verifier.py::test_gsm8k_hash_beats_intermediate_equation_*`.

### `rollout.deterministic_n` — leave it off

> ⚠ **Do not set `rollout.deterministic_n: true` globally** unless you
> specifically want bit-reproducible oracle tables for the paper. It loses
> throughput — every `n > 1` rollout call becomes `n` serial single-sample
> calls. The default (`false`) drops the per-request `SamplingParams.seed`
> on `n > 1` calls, matching verl / TRL / OpenRLHF convention, and is
> what every smoke-test / gate / oracle path in this repo actually needs.
> Engine-level reproducibility (`LLM(..., seed=cfg.rollout.seed)`) is
> always on regardless.

Background, when you need it: `vLLM 0.6.4.post1 + enable_prefix_caching=True
+ a fixed per-request SamplingParams.seed + n > 1 + instruct-style shared
prompt prefixes` is a known collapse mode — it was the root cause of the
original `rho-1b-sft-GSM8K` Day-1 gate returning `mean_group_reward_std=0.0`
on 256 / 256 groups. `VLLMRolloutRunner._sampling_params` enforces the fix
at the API boundary. See `src/grpocredit/rollout/vllm_runner.py`'s module
docstring for the full reasoning, and `SERVER2_RUNBOOK.md` §2.3 for the
operational consequences (exit code 7 from the diversity sentinel). Turn
`deterministic_n: true` on only when generating the final numbers you
intend to publish, and only via a tiny overlay YAML scoped to that one
config — never in `base_*.yaml`.

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
