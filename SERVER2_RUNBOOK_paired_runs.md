# Server2 runbook — Phase B1 (VinePPO + GRPO + oracle)

Single launch reference for the three paired goals:

1. **Reproduce VinePPO** training curves (in `VinePPO/`).
2. **Run GRPO** on the **same SFT'd init / same dataset / same trainer** so curves
   are directly comparable (in `VinePPO/`, treetune-internal GRPO; `grpocredit/`
   verl-GRPO is a separate, not-yet-built track — see §6).
3. **Q-variance oracle** on the first several GRPO iter checkpoints
   (in `grpocredit/`, calling the existing 4-step Phase-A pipeline).

Everything below assumes server2 (single 80 GB GPU is enough for any one phase;
multi-GPU just speeds it up). For deeper detail on each subsystem see
`VinePPO/SERVER2_RUNBOOK.md` and `grpocredit/SERVER2_RUNBOOK.md`.

---

## 0. One-time setup

Done once when you set up server2. Re-run only if env changes.

### 0.1 Clone

```bash
cd ~/work && git clone <your-fork-of-GRPO_mcts.git>
cd GRPO_mcts
```

### 0.2 Two conda envs

Hard rule: never cross-install. The two repos have incompatible torch/vllm pins.

```bash
# vineppo env (frozen baseline) — see VinePPO/requirements.txt
conda create -n vineppo python=3.10 -y
conda activate vineppo
conda env config vars set CUDA_HOME=$CONDA_PREFIX
conda deactivate && conda activate vineppo
pip install -r VinePPO/requirements.txt

# grpocredit env (oracle code) — see grpocredit/SERVER2_RUNBOOK.md §1.2
conda create -n grpocredit python=3.11 -y
conda activate grpocredit
conda env config vars set CUDA_HOME=$CONDA_PREFIX
conda deactivate && conda activate grpocredit
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124
pip install vllm==0.6.4.post1
pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
cd grpocredit && pip install -e ".[dev]" && cd ..
# optional pin-for-pin reproducibility:
# pip install --upgrade -r grpocredit/grpocredit-server2-pip-freeze.txt
```

### 0.3 Pre-download checkpoints

The two we need for the rho-1b GSM8K run are tiny; the 7B is for the headline.

```bash
huggingface-cli download realtreetune/rho-1b-sft-GSM8K            # 3 GB — required
huggingface-cli download realtreetune/deepseekmath-7b-sft-GSM8K   # 14 GB — for 7B run
```

**Critical invariant**: both are **SFT'd** checkpoints. **Never** RL on top of
`Qwen2.5-Math-7B-Instruct` — Qwen already GRPO'd it; that would be RL-on-RL.

### 0.4 wandb

```bash
wandb login                                # interactive once
export WANDB_API_KEY=...                   # or WANDB_MODE=offline (sync after)
```

---

## 1. Pre-flight sanity check

Run this **before** every fresh launch session. Fails fast on env / test / cache problems before you burn GPU hours.

```bash
cd ~/work/GRPO_mcts
bash grpocredit/scripts/sanity_check_server2.sh
```

Expected last line: `[sanity] PASS — server2 looks ready.`

What it checks:
- ≥1 GPU visible
- both conda envs exist
- `vineppo`: imports torch/vllm/deepspeed/transformers, runs `pytest tests/test_grpo_advantage.py -q` → 8 passed
- `grpocredit`: imports torch/vllm/sentence-transformers/math_verify, `load_config` smoke, full `pytest -q` → 42 passed
- HF cache: rho-1b ckpt present (required), 7B ckpts checked (warn-only)

If anything fails: fix root cause, do not proceed.

---

## 2. Goal 1 — VinePPO reproduction curve

Training curves of VinePPO at the published settings, starting from
`realtreetune/rho-1b-sft-GSM8K` (or 7B SFT'd for headline).

```bash
cd ~/work/GRPO_mcts/VinePPO
conda activate vineppo
export WANDB_PROJECT=GRPO_mcts-vineppo

# Fast 1.4B run — flushes any infra issue in ~few hours.
bash scripts/launch_server2_vineppo.sh rho1bSft2 GSM8K

# 7B headline (overnight on a single 80 GB GPU; faster on 8×).
# bash scripts/launch_server2_vineppo.sh deepseekSft2 GSM8K
```

**Outputs** (per-launch, paths relative to `VinePPO/`):

```
experiments/vineppo_rho1bSft2_GSM8K/
├── checkpoints/ckpt--iter_NNNN--*/hf_pretrained/   # per-iter HF dumps (no tokenizer files)
├── episodes__iterNNNN/
└── smoke_run_<UTC>.log
```

**W&B**: project `GRPO_mcts-vineppo`. Headline curves to watch on the run page:
`train/policy_loss`, `train/kl`, `train/value_loss`, `eval/accuracy`,
`episodes_metric/value_estimation_*` (advantage variance from the MC value path).

**Looks-good check**: by iter 2-3, `eval/accuracy` should be trending up vs the
SFT baseline; `train/kl` stays bounded (KL coef is 0.0001 from `refKl0.0001.jsonnet`).

---

## 3. Goal 2 — GRPO curve (matched trainer / KL / SFT'd init)

Same trainer, same KL, same SFT'd init, same dataset as §2 — only the
advantage formula differs. Treetune detects `Episode.advantages` and skips
GAE / value loss entirely (see `ppo_trainer.py:727`).

```bash
cd ~/work/GRPO_mcts/VinePPO
conda activate vineppo
export WANDB_PROJECT=GRPO_mcts-grpo

bash scripts/launch_server2_grpo.sh rho1bSft2 GSM8K
# bash scripts/launch_server2_grpo.sh deepseekSft2 GSM8K   # 7B variant
```

**Starting-point receipts** (verify with `grep`):
- `polIter_rho1bSft2_grpo_GSM8K.jsonnet:4` → `realtreetune/rho-1b-sft-GSM8K` (SFT'd)
- `polIter_deepseekSft2_grpo_GSM8K.jsonnet:4` → `realtreetune/deepseekmath-7b-sft-GSM8K` (SFT'd)

**Outputs**:

```
experiments/grpo_rho1bSft2_GSM8K/
├── checkpoints/ckpt--iter_NNNN--*/hf_pretrained/
├── episodes__iterNNNN/
└── smoke_run_<UTC>.log
```

**W&B**: project `GRPO_mcts-grpo`. Same metric set as VinePPO except the
critic-related ones (`valnet_*`) are absent (no critic in GRPO). New
`episodes_metric/grpo_*` keys: `grpo_dropped_zero_variance_frac` (groups
where every rollout got the same reward — no learning signal),
`grpo_n_total_groups`, `group_reward_mean/mean`, `group_reward_std/mean`.

**Looks-good check**:
- `episodes_metric/grpo_dropped_zero_variance_frac` < ~0.5 by iter 2
  (else policy is too saturated → check starting point isn't already RL'd)
- `eval/accuracy` rising
- `train/kl` bounded (same KL coef as VinePPO, so the two should be on the same KL scale)

---

## 4. Goal 3 — Q-variance oracle on GRPO iter checkpoints

Three oracle runs, escalating in coverage:

### 4.1 Iter -1 baseline (the SFT'd starting policy)

```bash
cd ~/work/GRPO_mcts/grpocredit
conda activate grpocredit
export WANDB_PROJECT=grpo-voi
bash scripts/run_oracle.sh configs/oracle/rho1b_sft_gsm8k.yaml
```

Output: `experiments/oracle/rho1b_sft_gsm8k/{day1_*, emb_var_*, oracle_*, GATE_REPORT.md}`.
Wall-clock: ~30 min on rho-1b. Captures the κ / ρ / position-curve baseline
**before any GRPO updates** — this is your iter -1 reference.

### 4.2 Iter N oracle on a GRPO checkpoint

After §3 has emitted at least `iter_0001`, oracle each iter you care about:

```bash
cd ~/work/GRPO_mcts/grpocredit
conda activate grpocredit
export WANDB_PROJECT=grpo-voi

# Iter numbering is 1-based. iter_0001 = policy after 1 GRPO iter.
# (There is no iter_0000 — that's the SFT'd init in §4.1.)
for n in 1 2 3 4 5; do
    bash scripts/run_oracle_on_grpo_iter.sh rho1bSft2 GSM8K $n
done
```

Output per iter: `experiments/oracle/grpo_iter/rho1bSft2_GSM8K/iter_NNNN/{day1_*, emb_var_*, oracle_*, GATE_REPORT.md}`.
Wall-clock: ~30 min per iter on rho-1b, ~3-4 h per iter on 7B.

**What the wrapper does**:
1. Resolves checkpoint at `VinePPO/experiments/grpo_rho1bSft2_GSM8K/checkpoints/ckpt--iter_NNNN--*/hf_pretrained/`.
2. Generates a `/tmp` overlay YAML that `extends:` the base oracle config and overrides:
   - `model.name_or_path` → local checkpoint path (so vLLM samples from the **GRPO-trained** policy at iter N, not the original SFT'd init)
   - `model.tokenizer_name_or_path` → original Hub repo (the GRPO trainer saves weights only, no tokenizer files)
   - `output_dir`, W&B group/job_type
3. Hands off to `scripts/run_oracle.sh`.

**Override knobs**:
- `N_TRAJ=50 bash scripts/run_oracle_on_grpo_iter.sh ...` — cheaper oracle (50 traj instead of 100)
- `GRPO_RUN_DIR=/abs/path/to/run bash scripts/...` — if you set `APP_DIRECTORY` at GRPO launch
- `VINEPPO_DIR=/abs/path/to/VinePPO bash scripts/...` — if the repo layout differs

### 4.3 Cross-iter comparison

After §4.1 + §4.2 finish, assemble a per-iter table:

```bash
cd ~/work/GRPO_mcts/grpocredit
echo "iter | kappa | rho_H_fwd | rho_s2 | MI_concord | group_var_frac"
for d in experiments/oracle/rho1b_sft_gsm8k experiments/oracle/grpo_iter/rho1bSft2_GSM8K/iter_*; do
    iter=$(basename "$d")
    kappa=$(cat "$d/oracle_kappa.txt" 2>/dev/null | head -1 | awk '{print $1}')
    echo "$iter | $kappa | (see oracle_correlations.json for rho_H_fwd, rho_s2) | (see emb_var_summary.json) | (see day1_group_variance.json)"
done
```

The κ trajectory across iters is itself a paper figure (whether VoI's
selection headroom changes as the policy trains).

---

## 5. End-to-end paired-run wall-clock budget

Single 80 GB GPU, rho-1b GSM8K (everything cheap):

| Step | Wall-clock |
|---|---|
| §1 sanity check | ~3 min |
| §2 VinePPO 650-iter run | ~4-6 h (depends on rollout count) |
| §3 GRPO 650-iter run | ~3-5 h (no value-estimation pass → faster than VinePPO) |
| §4.1 SFT-init oracle | ~30 min |
| §4.2 5 iter oracles × 30 min | ~2.5 h |
| **Total** | **~10-14 h** |

7B headline (deepseekSft2): roughly 4-6× the time. Budget overnight.

---

## 6. Known caveats

- **`grpocredit/training/` (verl-based GRPO) is empty.** Treetune-GRPO is the
  GRPO used here — same trainer as VinePPO so curves are bit-for-bit
  matched-framework. The plan's Phase B2a (verl GRPO + VoI hooks) is a
  separate sprint; not built.
- ~~**rho-1b prompt template drift**~~ **RESOLVED.** All three VinePPO-SFT'd
  oracle configs (`rho1b_sft_gsm8k.yaml`, `deepseek_math_sft.yaml`,
  `deepseek_math_sft_gsm8k.yaml`) now use `prompt_template: vineppo_math_task`
  (`"[MATH_TASK] Problem:\n{q}\n\nSolution:"`) + VinePPO's eval-time
  sampling (`temperature: 0.35`, `top_p: 0.9`, `stop: ["\n\n\nProblem:"]`) —
  exactly matching the template those checkpoints were SFT'd on and the
  template VinePPO's GRPO trainer continues to train under. This also
  makes §4.2 iter-oracles distribution-aligned automatically, since
  `run_oracle_on_grpo_iter.sh` generates an overlay that `extends:` the
  base oracle YAML and only replaces `model.name_or_path`. See
  `SERVER2_RUNBOOK.md` §2.2 (RESOLVED) and
  `tests/test_prompt_templates.py` for the pinning tests.
- **Iter checkpoints are 1-based**: there is no `iter_0000`. iter -1 = SFT'd
  init (§4.1); iter 1..N = §4.2. The wrapper lists available dirs on miss.
- **HF dump has no tokenizer**: VinePPO's `_save_hf_pretrained` saves model
  weights only. The wrapper pins `tokenizer_name_or_path` to the original
  Hub repo. If you supply a custom `[base_oracle_yaml]` 4th arg, you're
  responsible for the tokenizer mapping (`TOKENIZER_REPO=...` env-var).
- **MATH path on rho-1b is broken**: `polIter_rho1bSft2_ppo_MATH.jsonnet:1`
  hardcodes `realtreetune/rho-1b-sft-MATH`, which is **not published**. Only
  GSM8K is available for rho-1b. Use deepseekSft2 for MATH.
- **NCCL overlay corruption**: if `nvidia-smi` shows the right driver but a
  collective fails with `CUDA driver version is insufficient for CUDA
  runtime version`, run:
  `pip install --force-reinstall --no-deps nvidia-nccl-cu12==<version pip already records>`.
- **Concurrent oracle + GRPO on the same GPU**: oracle vLLM uses
  `gpu_memory_utilization: 0.9` by default. To run them concurrently, drop
  to ≤0.45 in the relevant oracle YAML, or use a second GPU, or serialize
  (recommended — cleaner).

---

## 7. Files this runbook references

| Path | Role |
|---|---|
| `VinePPO/scripts/launch_server2_vineppo.sh` | Goal 1 launcher |
| `VinePPO/scripts/launch_server2_grpo.sh` | Goal 2 launcher |
| `VinePPO/configs/polIter_*_grpo_*.jsonnet` | GRPO configs (4 model×task combos) |
| `VinePPO/src/treetune/episode_generators/math_grpo_episode_generator.py` | GRPO advantage compute + episode build |
| `VinePPO/configs/trainers/grpo_MATH.jsonnet` | trainer overlay (no critic, no whiten) |
| `VinePPO/tests/test_grpo_advantage.py` | 8-test correctness gate (`PYTHONPATH=src pytest tests/test_grpo_advantage.py -q`) |
| `grpocredit/scripts/run_oracle.sh` | Phase-A oracle (Day 1+2A+2B+3) |
| `grpocredit/scripts/run_oracle_on_grpo_iter.sh` | Goal 3 wrapper (per-iter oracle) |
| `grpocredit/scripts/sanity_check_server2.sh` | §1 pre-flight gate |
| `grpocredit/configs/oracle/*.yaml` | per-policy oracle configs |
