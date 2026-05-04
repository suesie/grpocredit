# Sprint Gate Report

**Overall:** `wrong_starting_policy` (exit_code=6)

## Gates

| Check | Value | Threshold | Status |
|---|---|---|---|
| Group-variance fraction at step 0 (sft_warmup_plan §5) | 0.492 | ≥ 0.5 | fail |
| Concordance: top-1 agreement (per-traj) | 0.561 | ≥ 0.50 | pass |
| Concordance: κ_emb (selection concentration) | 1.651 | ≥ 1.50 | pass |
| κ (variance concentration) | 1.762 | ≥ 3.0 | fail |
| ρ(s_2, Var(Q^π)) 95% CI low | nan | ≥ ρ_gate = 0.615 | fail |
| Position curve | `bimodal` | mid-peak preferred | — |

## Next steps

Group-variance gate FAILED at step 0 — starting policy is too saturated or too weak (sft_warmup_plan.md §5). DO NOT proceed to RL. Switch `π_ref` to a SFT-warmed model (Option B: `realtreetune/deepseekmath-7b-sft-{MATH-v2,GSM8K}`; or Option A: SFT `Qwen2.5-Math-7B` base ourselves). Re-run sprint Day 1+.

## Raw values

- group-variance fraction (informative): 0.4921875
- **Concordance selection metrics (config A: K=4/len=30):**
  - top-1 agreement: 0.5606060606060606
  - κ_emb: 1.6507818711900344
  - ρ(emb_var, reward_var) Spearman (legacy): 0.29817794277430776
  - Spearman 95% CI: [0.1753137280006374, 0.4118967712040947]
- **Concordance selection metrics (config B: K=8/len=15):**
  - top-1 agreement: 0.48484848484848486
  - κ_emb: 1.3627943485086342
  - ρ(emb_var, reward_var) Spearman (legacy): 0.15956479931323028
- κ: 1.7615264373661068
- **Day 2B selection metrics (per-trajectory, vs Var(Q^π)):**
  - H_fwd: top1=0.5064935064935064, κ_sig=1.2887090873806895
  - H_token: top1=0.5844155844155844, κ_sig=1.6090163447448202
- Spearman correlations (legacy):
  - ρ(s_2, Var(Q^π)): nan
  - ρ 95% CI: [nan, nan]
  - ρ(H_fwd, Var(Q^π)): 0.15494177274650434 (K=10)
  - ρ(H_fwd) 95% CI: [0.0362361173087157, 0.2693359411125515]
  - ρ(H_token, Var(Q^π)): 0.191210807593351
  - ρ(H_token) 95% CI: [0.07351292063360372, 0.3036554698547252]
- ρ_gate: 0.6151907371101499
- position shape: bimodal
