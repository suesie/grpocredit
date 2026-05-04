


Day 2 A and Day 2B now use informative groups(not all fail or all success within the group) only.

Day 2A: We found clustering (of lookahead embeddings) a poor indicator overall and will not use it. Instead, we use the embedding variance.
Day 2A — Spearman ρ is the wrong metric. The cascade's decision is per-trajectory top-k selection (pick ~2 of ~2–8 boundaries), not global rank correlation. Spearman pools within- and across-trajectory variation; what we actually need is *"within this trajectory, does emb_var pick the same boundary as reward_var?"*

Replace ρ with three per-trajectory metrics:

- **(a) top-1 agreement** — fraction of trajs (≥ 2 boundaries) where argmax emb_var = argmax reward_var. Answers: "if we could probe one boundary per traj, does lookahead pick the right one?"
- **(b) overlap@2** — Jaccard between top-2 by emb_var and top-2 by reward_var on trajs with ≥ 4 boundaries. Mirrors the cascade's actual top-`f_sel` selection.
- **(c) κ_emb** — reward_var at the argmax-emb_var boundary ÷ mean reward_var across boundaries in that traj. Direct estimate of the variance-concentration factor from the embedding signal; > 1 means selection value even if ρ is low.

Add a Config B run: lookahead at #rollout=8/len=15, terminal reward variance still at #rollout=4. → ρ ≈ 1.1, very small.

Day 2 B Entropy should not be based on cluser. But the average/discounted sum of entropy of next K tokens? Or the average/discounted sum of KL divergence of next K tokens to reference(teacher) distribution?





what should I do next?

● Based on where you are:

  1. Then inspect boundary_annotated_responses.txt — eyeball whether the boundary positions look sensible on actual GSM8K
  solutions. Check emb_var_summary.json for the ρ values. On rho-1b, expect rho_cosine_long_only = NaN (too few long
  boundaries) and an inflated overall ρ due to the short-CoT / lookahead-length issue.
  2. Run on the headline policies (DeepSeekMath-SFT) — these have longer CoTs, more boundaries, and the 30-token lookahead
  will be a genuine preview. This is where the ρ result actually matters:
  bash scripts/run_oracle.sh configs/oracle/deepseek_math_sft.yaml
  bash scripts/run_oracle.sh configs/oracle/deepseek_math_sft_gsm8k.yaml
  3. Read the GATE_REPORT.md from DeepSeekMath runs. Three outcomes:
    - ρ ≥ 0.3: Embedding diversity predicts reward diversity. Stage 2 clustering is validated as a VoI proxy. Proceed with the
   main plan.
    - 0.15 ≤ ρ < 0.3: Marginal. Consider using continuous emb_var_cosine directly as the VoI signal instead of binary H_sem
  clustering.
    - ρ < 0.15: Lookahead embeddings don't predict reward variance. Either the 30-token window is too short for MATH problems,
   or sentence-T5 embeddings don't capture reward-relevant structure. Consider Plan B or a different embedding model.
  4. If ρ is strong, the next structural task is verl integration (the empty src/grpocredit/training/ directory) — wiring the
  VoI cascade into actual GRPO training as the custom advantage hook.


===============================



  ===========================

  TODO: teacher model, on-policy distillation.








## Alternative ways to trigger rollouts
(a) H_fwd as Stage 1.5 (between Stage 1 and Stage 2) — FREE pre-filter:

  Use H_fwd to pre-filter before running the expensive lookahead. If H_fwd at a boundary is low (policy is confident over the next K steps), the
  boundary is unlikely to have high Q-variance → skip the lookahead entirely. This turns Stage 2 into a confirmatory stage that only fires on the
  ~30% of boundaries where H_fwd is elevated, cutting the lookahead cost by 70%.

  (b) Just use Stage 1 (H_token × w_pos) and skip Stage 2 entirely:

  H_token alone got ρ = 0.116 with Var(Q^π). If H_fwd does better (say ρ ≈ 0.2–0.3), and the within-trajectory top-1 agreement is good, there may
  be no need for the embedding-based Stage 2 at all. The cascade simplifies to: Stage 0 (group filter) → Stage 1 (H_token × w_pos + H_fwd) →
  probe.


  Entropy signal: You're asking a fundamental design question. Three options on the table:

  | Signal | Cost | What it captures |
  |---|---|---|
  | `H_sem` (current. bad) | 4 rollouts × 30 tokens + sentence-T5 encode + clustering | Semantic diversity of continuations — but binary at K=4 |
  | `mean(H(π(·\|s_t)))` for `t = b..b+K` | Free — already in `trajectory.token_entropies` | Policy uncertainty over the next K tokens in the actual trajectory |
  | `mean(D_KL(π ‖ π_ref))` for `t = b..b+K` | Requires reference model forward pass | Drift from reference — the PPO/GRPO-relevant signal |


  #Q2.   Based on the 2A results, is the lookahead signal good enough to trigger the oracle? If not, what is the best way to trigger the oracle? Also, the lookahead length is 30, and the embedding model is sentence-t5-base, and we need to compute the senmantic variance. Does the cost make sense to detect the triggering boundary?



  What to look for in the output:
  1. oracle_correlations.json — compare ρ(H_fwd, Var(Q^π)) vs ρ(H_token, Var(Q^π)). On rho-1b with short CoTs (mean ~2.3 boundaries), H_fwd may
  not gain much over H_token since K=10 often captures most of the remaining response. The real test is on DeepSeekMath.
  2. oracle_summary.json — n_informative_prompts should be ~126 (instead of the old 100 random trajectories). κ computed from informative groups
  should be more meaningful.
  3. oracle_kappa.txt — selection_score should now show h_fwd (or s2 if H_fwd collapsed to constant).

  Then — the headline run that actually matters:

  bash scripts/run_oracle.sh configs/oracle/deepseek_math_sft.yaml
  bash scripts/run_oracle.sh configs/oracle/deepseek_math_sft_gsm8k.yaml

  DeepSeekMath-SFT has longer CoTs (~8–15 boundaries/trajectory), so:
  - H_fwd over K=10 tokens is a genuine multi-step preview (not the whole remaining response)
  - More boundaries per trajectory → tighter CIs on all correlations
  - The concordance diagnostic (Day 2A) long-only subset will have enough samples

  These are the headline RL-init policies. Their GATE_REPORT.md determines whether VoI proceeds or pivots to Plan B.

  ========================


============================

● Full Pipeline Analysis (rho-1b, latest run)

  This is the complete end-to-end run with all our changes. Let me walk through what the numbers mean and what stands out.

  ---
  Day 2A — Embedding-variance diagnostic

  Already analyzed, but noting for completeness: Config A passes (top1=0.561, κ_emb=1.651). Config B fails.

  Day 2B — Oracle with selection metrics

  This is the critical new data. 288 oracle records from 126 informative prompts (vs 213 from 100 random before).

  Selection metrics:

  ┌─────────┬───────┬───────┬───────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Signal  │ top1  │ κ_sig │ overlap@2 │                                              Assessment                                              │
  ├─────────┼───────┼───────┼───────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ H_token │ 0.584 │ 1.609 │ 0.573     │ Best signal                                                                                          │
  ├─────────┼───────┼───────┼───────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ s2      │ 0.558 │ 1.494 │ 0.573     │ Competitive (but s2 = H_token × H_sem, and H_sem is binary → s2 ≈ H_token where H_sem=1, 0           │
  │         │       │       │           │ elsewhere)                                                                                           │
  ├─────────┼───────┼───────┼───────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ H_fwd   │ 0.506 │ 1.289 │ 0.360     │ Weakest — barely above random                                                                        │
  └─────────┴───────┴───────┴───────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────┘

  H_token beats H_fwd. This is the key finding. Single-token entropy at the boundary outperforms the average entropy over the next 10 tokens as a
  predictor of Var(Q^π). This was unexpected but makes sense for rho-1b:

  1. Short CoTs (mean ~2.3 boundaries). With only ~2 boundaries per trajectory, most trajectories have a "computation boundary" and an "answer
  boundary." The single-token H at the computation boundary (where the model decides e.g. "multiply" vs "divide") is the load-bearing decision.
  Averaging over the next 10 tokens dilutes this — the 9 tokens after the decision point are execution of a choice already made.
  2. H_fwd K=10 captures too much noise. On a 50-token CoT, 10 tokens is 20% of the response. Many of those tokens are deterministic (writing
  digits, formatting). Their low entropy drags down H_fwd at genuinely uncertain boundaries.
  3. H_fwd overlap@2 = 0.360 is notably worse than H_token's 0.573. When the cascade has to pick 2 boundaries from ≥4, H_fwd's multi-step
  averaging muddies the ranking.

  Spearman vs selection metrics — the gap is dramatic:

  H_token gets ρ = 0.191 (Spearman) but top1 = 0.584 and κ_sig = 1.609. Spearman makes it look like a weak signal; the selection metrics show it's
   actually a good selector — picking boundaries with 61% more Var(Q^π) than the trajectory average.

  H_sem is negatively correlated (ρ = -0.128). The binary clustering gate is actively anti-correlated with Var(Q^π). Boundaries where all 4
  lookaheads cluster together (H_sem=0) actually have higher Q-variance than multi-cluster boundaries. This kills the Stage 2 design on rho-1b —
  the cosine-clustering gate is worse than useless.

  Day 3 — Gate decisions

  ┌─────────────────┬─────────────────────┬────────┬─────────────────────────────────────────────────────────────────┐
  │      Gate       │        Value        │ Status │                             Comment                             │
  ├─────────────────┼─────────────────────┼────────┼─────────────────────────────────────────────────────────────────┤
  │ Group-variance  │ 0.492               │ FAIL   │ Expected for rho-1b debug policy                                │
  ├─────────────────┼─────────────────────┼────────┼─────────────────────────────────────────────────────────────────┤
  │ Concordance     │ top1=0.561, κ=1.651 │ PASS   │ Embedding lookahead works for boundary selection                │
  ├─────────────────┼─────────────────────┼────────┼─────────────────────────────────────────────────────────────────┤
  │ κ               │ 1.76 [0.66, 2.99]   │ FAIL   │ Below 3.0 but improved from 1.25 → 1.76 with informative groups │
  ├─────────────────┼─────────────────────┼────────┼─────────────────────────────────────────────────────────────────┤
  │ ρ(s₂, Var(Q^π)) │ NaN                 │ FAIL   │ s₂ collapsed — H_sem=0 everywhere                               │
  └─────────────────┴─────────────────────┴────────┴─────────────────────────────────────────────────────────────────┘

  κ = 1.76 (was 1.25 in the prior run, and 2.16 in the previous re-run). The variation across runs (1.25 → 2.16 → 1.76) is due to bootstrap CI
  width [0.66, 2.99] — the true κ on rho-1b is genuinely uncertain with only ~288 records.

  position_shape = bimodal (was mid_peak, then flat in different runs). Again, small sample with short CoTs → the shape classification is
  unstable.

  ---
  Implications for the method

  1. Stage 2 (cosine clustering) is broken and should be bypassed on short-CoT models. H_sem is anti-correlated. The binary gate would filter out
  the best boundaries. On rho-1b, the cascade should be: Stage 0 (group filter) → Stage 1 (H_token × w_pos) → probe. No Stage 2.

  2. H_token alone is a viable VoI selector. κ_sig = 1.609 means probing the highest-entropy boundary per trajectory gives 61% more outcome
  variance than random. This is the voi_stage1 config.

  3. H_fwd doesn't help on short CoTs. K=10 is too wide a window for 50-token responses. Two options:
  - Reduce K to 3–5 for short-CoT models (auto-scale K = min(10, remaining_tokens // 5))
  - Accept that H_fwd is a signal for long-CoT models only, and the DeepSeekMath run is the real test

  4. The rho-1b gate outcome is correct: exit 6, wrong starting policy. Every metric confirms this is too small/short-CoT for a defensible VoI
  headline. But the selection metrics validate the measurement infrastructure — the signals work, just on a policy that doesn't have enough room
  for VoI to matter.
============================
============================
#Q1: Deep think: for Day 2 B result, H_token beats H_fwd.  Single-token entropy at the boundary outperforms the average entropy over the next 10
tokens as a predictor of Var(Q^π). I suspect it is due to the way of our computation of entropy or hyperparameters or other reasons.

#A1:
 Root causes: why H_fwd < H_token as a predictor

  Cause 1 (primary): On-policy conditional entropy is the wrong quantity

  H_fwd computes the average entropy along the realized path:

  H_fwd = (1/K) Σ_{k=0}^{K-1} H(a_k | s_b, a_0*, ..., a_{k-1}*)

  Once the model commits to a_0* at the boundary, the rest of the path is often deterministic execution. A boundary where the model hesitates
  between "multiply by 2" and "divide by 2" (H_token ≈ 1.0) but then deterministically computes "48 × 2 = 96\n" (H ≈ 0.05 for the next 9 tokens)
  gives:

  - H_token = 1.0 (correctly identifies the decision point)
  - H_fwd = (1.0 + 9 × 0.05) / 10 = 0.145 (diluted 7×)

  What we'd actually want is the entropy of the K-step marginal — sample many paths from s_b and measure their diversity. But that requires
  multiple rollouts, which is exactly what Stage 2 does (and what we're trying to avoid with H_fwd).

  Cause 2 (critical): logprobs=1 makes all entropy estimates terrible

  The vLLM runner at vllm_runner.py:169 defaults to logprobs=1. This means step_dict at each generation step contains only the sampled token. The
  "entropy" computed at line 278–284 is:

  H_estimated = -p_sampled × log(p_sampled)

  This is a single term of the entropy sum, not the entropy. For a high-confidence token (p=0.95), the estimate is 0.049 — close to the true
  entropy for a peaked distribution. But for a genuine decision point (p=0.5), the estimate is 0.347, while the true H over the full vocabulary
  could be 1.5+.

  This affects both H_token and H_fwd equally — both read from the same trajectory.token_entropies array. But H_token survives because:
  - It only uses position b (the boundary token), where the policy is making a real decision
  - Even with top-1 logprobs, the ranking of boundaries by -p*log(p) is monotone in p for p > 1/e ≈ 0.37 (most sampled tokens), so the ordering is
   roughly preserved
  - H_fwd averages over 10 positions, 9 of which have p_sampled ≈ 0.95+ (deterministic), contributing near-identical near-zero values that destroy
   the ranking

  Cause 3: K=10 is too wide for short CoTs

  With ~22 tokens between boundaries on rho-1b, K=10 extends 45% of the way to the next boundary. Tokens 5–9 may belong to a different reasoning
  step — their entropy is about a different decision than the one at position b.

  What to fix

  Fix 1 (high impact, zero cost): Increase logprobs to 5–10

  Change the default in _sampling_params from logprobs=1 to logprobs=5 (or make it configurable). This gives a much better entropy estimate —
  top-5 captures ~95% of probability mass for most distributions. The compute cost is negligible (vLLM already computes the full softmax;
  returning 5 logprobs vs 1 is free).

  This would improve both H_token and H_fwd, but H_fwd would benefit more because the entropy at non-boundary positions (where the sampled token
  has p ≈ 0.5–0.8, not p ≈ 0.95) would be estimated much more accurately.

  Fix 2 (medium impact): Use max instead of mean for H_fwd

  H_fwd_max = max(token_entropies[b : b+K])

  This finds the highest-entropy token in the window — the decision point — without diluting it with deterministic tokens. It's robust to the case
   where the true decision point is 2–3 tokens after the syntactic boundary (e.g., the boundary is at "\n" but the real decision is the first
  content token of the next step).

  Fix 3: Auto-scale K

  K = min(h_fwd_k, remaining_tokens // 3)

  Prevents the window from extending beyond the current reasoning step on short CoTs.

  Fix 4 (different approach): H_fwd as sum, not average

  The chain rule says Σ H(a_k | ...) = H(a_0, ..., a_{K-1} | s_b, path*). The sum is the joint entropy along the realized path. Using sum instead
  of mean doesn't lose signal through averaging. But the on-policy conditioning issue (Cause 1) still applies.

  What NOT to fix

  Don't try to make H_fwd compute the true multi-step marginal entropy — that requires sampling multiple paths, which is Stage 2's job. The right
  framing is: H_fwd is a cheap upper-bound on the signal, and if H_token already works as well or better, we should just use H_token and not pay
  the conceptual complexity tax of H_fwd.
 #Q1 follow-up:  Deep think: 1. if we modify vLLM runner at vllm_runner.py:169 defaults to logprobs=1. Will that impact the GRPO training? 2. Should we make logprobs even larger and K small
  2--5? 3. sum of entropy will lead to same result as mean of entropy right? 4. "don't make H_fwd compute the true multi-step marginal entropy". But since we have lookahead
  rollouts in Day 2A, can we compute that in Day 2A?

#Q2: You said "  The Stage 2 design assumes that short lookahead diversity predicts terminal reward diversity. But there are two ways this can fail:

  1. Lookahead is too short to capture decision-relevant divergence. On GSM8K with mean remaining ~78 tokens, 30 tokens is ~40% of the remainder —
   arguably enough. But on MATH with 300+ token remainders, 30 tokens is <10% and may not reach the point where algebraic paths diverge.
  2. Sentence-T5 embeddings don't capture reward-relevant diversity. Two CoTs can be lexically different but mathematically equivalent (different
  variable names, same algebra). Or the same lexically but diverge later (same first 30 tokens, different final computation). Sentence-T5 captures
   surface-level semantic similarity, not mathematical content." Then for what applications and dataset and models will our assumption work best?
#Q3: Could the hyperparameters lead to the poor results? For example the way of detecting boundaries. I need to check the boundary quality. For example the sampling temperature in Day 2 A lookahead. For example, how is the semantic variance in Day 2 A computed? Any better way to comput it?
#Q4: Assume in the future I want to select triggering points across the batch instead of within the trajectory. For example, a batch has 10 informative groups, and first 3 groups have very high semantic variance/or lookforward entropy at each boundary, while the remaining 7 groups have small ones. I will allocate most of monte carlo rollout to trajecotires in the first 3 groups. In this case, which metric should I compute? Spearman?

===========================
How to re-run

  # Re-run the full oracle pipeline on rho-1b (cheapest, ~30 min):
  PROCEED_ON_POLICY_GATE_FAIL=1 bash scripts/run_oracle.sh \
      configs/oracle/rho1b_sft_gsm8k.yaml

  This runs Day 1 → 2A → 2B → 3 in order. The Day 2A output will now show:

    [Config A: K=4 / len=30]  K_emb=4  K_rew=4  lookahead=30 tokens
      Selection metrics (per-trajectory):
        top-1 agreement:   0.XXX  (N trajs with ≥2 boundaries)
        overlap@2:         0.XXX  (N trajs with ≥4 boundaries)
        κ_emb:             X.XXX  (N trajs with reward_var > 0)
      Spearman (global, legacy):
        ρ(emb_var, reward_var) = 0.298  95% CI [0.175, 0.412]

  On rho-1b with mean 2.0 boundaries/traj, overlap@2 will have very few qualifying trajectories (needs ≥4 boundaries). The key numbers to watch
  are top1_agreement and κ_emb — if κ_emb > 1, the embedding signal selects higher-reward-variance boundaries than random, even if Spearman is
  noisy.

  =======================
  ### result analysis
  #### Day 2A

  **Config A (K=4, len=30) — passes the selection gate.**
  Spearman ρ = 0.298 understates it; the per-trajectory selection metrics:
  - **top-1 agreement = 0.561** vs random ≈ 0.50 (most trajs have 2 boundaries; n=3.5 avg → random ≈ 0.29). Above random, modestly.
  - **overlap@2 = 0.590** on the 13 trajs with ≥4 boundaries. Solid vs random 2/n ≈ 0.33–0.50.
  - **κ_emb = 1.651** — the headline. Probing the boundary picked by max emb_var yields 1.65× the reward-variance of a random boundary in the same trajectory.

  **Config B (K=8, len=15) — fails.** top-1 = 0.485 (below the 0.50 coin-flip), κ_emb = 1.363, ρ_long-only = 0.089. 15 tokens ≈ 19% of the mean remaining 78 tokens — too short. Config A's 30 tokens (~39%) reaches far enough to see whether the CoT is heading toward the right computation.

  **Long-only caveat.** Config A's ρ drops 0.298 → 0.144 once `remaining > 60`, so ~half the Spearman signal comes from near-terminal boundaries where the 30-token lookahead ≈ the rest of the response. With selection metrics this matters less — κ_emb = 1.651 is computed across all boundaries, and the cascade will naturally route probes to whichever boundaries carry the signal. The real risk is transfer: on DeepSeekMath (200+ token CoTs) 30 tokens is ~15% of remaining and the near-terminal free lunch disappears.

  **Sample-size caveats.** top-1 over 66 trajs, overlap@2 over only 13. κ_emb uses 49/66 trajs with ≥2 boundaries — 17 had reward_var = 0 at every boundary (always-right or always-wrong, uninformative).

  **Bottom line.** Config A clears κ_emb > 1.5 and top-1 > 0.50 → genuine selection value on rho-1b. Config B's shorter lookahead doesn't pay for itself. DeepSeekMath is the actual test of whether Stage 2 earns its place in the cascade.
