# grpocredit — context for Claude Code

This file is auto-loaded by Claude Code. Read it before doing anything in this repo.

## Project

VoI-allocated credit assignment for GRPO on single-turn math reasoning. Sparse
MC baselines at selected pivots, chosen by a cheap→expensive Value-of-Information
cascade; TD-style segment advantages over probed pivots. Target venue: ICLR 2027
(deadline ~2026-09-25). Primary rival: VinePPO (Kazemnejad et al. 2025) at
matched compute.

Design is frozen in two documents in the parent directory:
- `../research_proposal_grpo_clean.md` — method (3 modules)
- `../experiment_plan_grpo_voi.md` — timeline, sprint gates, compute budget

## Committed decisions (do not re-litigate)

| ID | Decision | Reason (from plan §0) |
|---|---|---|
| D1 | Matched-compute regime, 90 rollouts / trajectory | VinePPO's validated K=9 config would dominate us at matched-probes |
| D2 | Binary Stage-2 gate at K_LA=4 | Continuous H_sem at K_LA=4 too noisy; upgrade to K_LA=8 only if survivor counts force it |
| D3 | **SUPERSEDED — see `../research_plan/sft_warmup_plan.md`.** Headline RL init is `realtreetune/deepseekmath-7b-sft-{MATH-v2,GSM8K}` (Option B); Qwen2.5-Math-7B base + own SFT is the secondary modernization run (Option A). Qwen2.5-Math-7B-Instruct is retired as an RL start (already SFT+GRPO'd) and used only as a saturation-ceiling probe. | Original D3 picked Qwen-Instruct as primary; that conflates VoI's contribution with the Qwen team's prior GRPO. New plan keeps the matched-compute story honest. |
| D4 | GSM8K + MATH joint train; AIME24 + OlympiadBench + MATH-500 OOD | Verifier-scorable; cascade Stage 0 needs verifiable answers |
| D5 | verl (Volcengine) as the RL framework | Cleanly supports custom advantage hooks; TRL is too rigid |

## Sprint gate thresholds (`scripts/sprint_d3_gate_report.py`)

| Check | Pass | Marginal | Fail |
|---|---|---|---|
| Group-variance fraction at step 0 (`sft_warmup_plan.md` §5) | `≥ 0.5` | (no marginal band) | `< 0.5` → wrong starting policy; switch `π_ref` (exit code 6) |
| ρ(emb_var, reward_var) — concordance | `≥ 0.3` | `0.15`–`0.3` | `< 0.15` → Plan B (§8 of plan) |
| `ρ(s_2, Var_{a~π}(Q^π))` 95% CI lower bound | `≥ ρ_gate` | straddles `ρ_gate` | below |
| κ | `≥ 3` | `2`–`3` | `< 2` (paper pivots to efficiency headline) |

`ρ_gate = sqrt(f_target / (f_sel · κ))` with `f_target = 0.10`, `f_sel = 0.15`.

Plan B (§8 of `experiment_plan_grpo_voi.md`) is pre-designed — if embedding-variance
diagnostic fails (lookahead diversity does not predict reward diversity), pivot to
doubly-robust GRPO with IG-as-implicit-baseline. ~70% infra overlap; only Stage 2
swaps out.

## Repo layout

```
src/grpocredit/
  common/      configs (pydantic + YAML extends), wandb wrapper, shared dataclasses
  rollout/     vLLM runner (full / continue / forced-first-token), boundary detector,
               math_verify wrapper, GSM8K/MATH/AIME24/Olympiad/MATH-500 loaders
  voi/         Stage 0 group filter, Stage 1 H_token·w_pos + H_fwd (multi-step entropy),
               Stage 2 sentence-T5 clustering, CUSUM auxiliary,
               cascade orchestrator (offline + online paths)
  advantage/   TD-style segment GAE over sparse pivots, James–Stein / SE shrinkage
  oracle/      Q^π-variance oracle (top-M forced actions + optional tail stratum),
               embedding-variance diagnostic, κ with bootstrap CI, position-decile curve,
               H_fwd (avg entropy over next K tokens) as multi-step VoI signal
  training/    (empty — verl integration lands in Week 1 of main phase)

scripts/
  sprint_d1_infra_smoke.py     Day 1 — infra smoke test
  sprint_d2_concordance.py     Day 2A — embedding-variance diagnostic
  sprint_d2_oracle.py          Day 2B — Q^π-variance oracle on informative groups
                               (group-variance probe → informative prompts →
                               H_fwd + H_token + H_sem + s_2 correlations with
                               Var(Q^π) → κ + position curve)
  sprint_d3_gate_report.py     Day 3 — decision table → GATE_REPORT.md
  run_sprint.sh                runs all four in order

configs/
  base_qwen_math.yaml               base (all others `extends:` this)
  baselines/{grpo,vineppo,random_cascade}.yaml
  proposed/{voi_stage1,voi_full,voi_full_cusum}.yaml
```

## Conventions

- **Wandb is mandatory.** Every executable script wraps in `init_wandb(...)` →
  `wandb_run.finish()`. Metrics + configs + key JSON/CSV artifacts go up. JSON/CSV
  under `experiments/` is a secondary record, not a replacement. On air-gapped
  nodes: `WANDB_MODE=offline`, sync after. Default project: `grpo-voi`.
- **Configs use `extends:`** — every variant inherits from `base_qwen_math.yaml`
  and overrides only the differing keys. `load_config()` handles the merge.
- **Boundary indexing.** `Boundary.token_position = k` means prefix is
  `trajectory.token_ids[:k]` and the next-token distribution at `s_b` sampled
  `token_ids[k]`. `token_entropies[k]` is `H(π(·|s_b))`.
- **Stage-2 lookahead dicts are keyed by `id(boundary)`**, not by
  `boundary_idx` — `boundary_idx` is only unique within one trajectory, so
  dict merges across a group would collide.
- **Default advantage shrinkage** is James–Stein `α(M) = M/(M+4)`. The
  `shrinkage.mode = "none"` path is for VinePPO-style uniform K that doesn't
  need it.
- **Verifier** is the `math-verify` PyPI package, not a VinePPO vendor.
  Wrapped in `grpocredit.rollout.verifier.MathVerifier`, which runs a
  **priority-ordered registry** of extractors in
  `_EXTRACTORS: list[(method_tag, extractor_fn)]`. Current order
  (most-authoritative first, pinned by `test_extractor_registry_order`):

      gsm8k_hash (####)  →  answer_tag (<answer>X</answer>)
                         →  boxed (\boxed{X})
                         →  answer_is ("the answer is X")
                         →  fallback (last numeric token)

  **Priority ordering is load-bearing.** The pre-fix priority had `####`
  in slot 3 behind a `=\s*<value>` heuristic; that cost rho-1b-sft-GSM8K
  an empirical 5× pass@1 under-reporting because every CoT `=` step beat
  `####`. If you touch `_EXTRACTORS`, run `pytest tests/test_verifier.py
  -v` and inspect an actual `day1_rollouts.jsonl` via
  `scripts/inspect_day1_rollouts.py` — do not trust the aggregate alone.

  **New model / new dataset** — if rollouts use a convention that isn't
  in the current registry (e.g., R1-distilled with only `<answer>` tags,
  or a benchmark whose answer marker isn't in this list), you MUST add
  an extractor before running oracle/gate. Check: `scripts/inspect_day1
  _rollouts.py --aggregate` → if `verifier extract method` is dominated
  by `fallback`, the verifier is failing silently. Workflow in
  `SERVER2_RUNBOOK.md` §2.4 ("Extending the verifier").

  **Not covered** — multiple-choice benchmarks, code-generation, proofs:
  write a new `XXXVerifier` class with the same `score(response,
  ground_truth) -> VerifierResult` contract rather than extending the
  registry. The oracle pipeline only depends on that interface.

## Running the sprint

```bash
# Assumes `pip install -e ".[vllm,dev]"` already done, WANDB_API_KEY exported.
export WANDB_PROJECT=grpo-voi
bash scripts/run_sprint.sh
# Exit codes from sprint_d3: 0 proceed · 2 proceed-w-caveats · 3 pilot-required
# · 4 pivot to Plan B · 5 missing data · 6 wrong starting policy
```

## Known landmines

- **vLLM on Windows** — doesn't install. Development is on a Linux GPU node.
- **Offset mapping** — `scripts/_shared.offset_mapping_from_tokenizer` primary
  path uses `return_offsets_mapping=True` (fast tokenizers). The greedy-decode
  fallback is O(n²) and should rarely be hit.
- **`id()` identity for boundaries** — reused across stages 1→2→cascade. Don't
  `copy.deepcopy` a boundary then look it up in a lookahead dict.
- **Tail stratum sampling** — uses rejection against the top-M token set, not
  true conditional sampling. Matches plan §3.1.3's operational rule; mention
  the caveat if reviewers ask.

## Open items (as of sprint start, 2026-04-24)

- [ ] verl fork with a custom advantage hook (Day 4 of sprint / Week 1 of main)
- [ ] Decide Stage-1 `w_pos_shape`: tent default, but if the oracle position
      curve is flat, switch to `uniform`; if bimodal, switch to `lookup` with
      the `position_lookup.csv` the oracle writes.
- [ ] CUSUM stacking — only if `ρ(|δ_t|, Var(Q^π)) > ρ(H_token, Var(Q^π))`
      in the offline diagnostic. Default config has CUSUM off.
