# Server2 runbook — grpocredit phase rollout

This document is the single launch reference for everything the user runs on
the powerful-GPU machine ("server2"). It is split into three phases. Each
phase is independently launchable.

| Phase | What it produces | Status | Where to look |
|---|---|---|---|
| **A — Offline Q^π-variance oracle** on 4 starting policies | `experiments/oracle/{policy}/GATE_REPORT.md` × 4, plus per-policy oracle / concordance / position-curve / **group-variance** artefacts. Decides whether to proceed with VoI cascade or pivot to Plan B. | **Ready to run** (this document) | §1, §2 |
| **B1 — VinePPO + GRPO inside treetune** | Paired training runs, same model + dataset, wandb-logged accuracy / KL / advantage-variance curves, directly comparable. | **Ready to run** | `VinePPO/SERVER2_RUNBOOK.md` |
| **B2a — verl GRPO baseline + VoI on DeepSeekMath SFT'd** | Plain verl-GRPO vs verl-GRPO+VoI on `realtreetune/deepseekmath-7b-sft-{MATH-v2,GSM8K}`, matched compute. **Headline experiment** per `sft_warmup_plan.md` §4.1. | Code lands in turn after this | §4 of this doc |
| **B2b — Generalization run on own-SFT'd Qwen-base** | SFT `Qwen2.5-Math-7B` (base) ourselves per `sft_warmup_plan.md` §3.A, then run GRPO + VoI on top. Demonstrates the gain replicates on a modern base. | Code lands after B2a | §5 of this doc |

> **Methodology anchor.** Phase A and the entire B-series are scoped against
> `research_plan/sft_warmup_plan.md`. That document supersedes D3 of
> `experiment_plan_grpo_voi.md`: `Qwen2.5-Math-7B-Instruct` is **not** an RL
> starting point (it's already SFT+GRPO'd by Qwen with `Qwen2.5-Math-RM-72B`),
> only a saturation-ceiling probe. Headline RL init is VinePPO's published
> `realtreetune/deepseekmath-7b-sft-{MATH-v2,GSM8K}`.

---

## 0. Server2 prerequisites

* NVIDIA GPU with CUDA Compute Capability ≥ 7.0 (Ampere, Hopper, or newer recommended). Phase A oracle runs comfortably on a single 80 GB GPU; vLLM uses ~50 GB at `gpu_memory_utilization=0.9` for Qwen-Math-7B-Instruct.
* CUDA driver ≥ 525 (anything supporting CUDA 12.4 runtime works).
* `git`, `conda` (or `mamba`), ≥ 50 GB free disk for env + 3 model checkpoints.
* `WANDB_API_KEY` exported, or `WANDB_MODE=offline` for air-gapped nodes.
* Network access to Hugging Face Hub (or pre-downloaded checkpoints — see §1.3).

### 0.1 Login node vs. compute node (SLURM)

On the shared AWS cluster (`shared-aws-usw1-1` as of this writing), **login
nodes do not have GPU drivers installed**. Running vLLM directly on the
login node fails with `libcuda.so.1: cannot open shared object file` → vLLM
then raises `RuntimeError: Failed to infer device type`. Every launch
command in this runbook MUST be executed on a compute node.

#### 0.1.1 See which QoS / accounts you're allowed to use

Before any `salloc` — always useful to double-check your permissions didn't
change under you:

```bash
# Full list of (account, QoS) pairs you're authorised against. `%30`/`%150`
# just widen the column so long QoS names don't truncate.
sacctmgr show assoc format=user,account%30,qos%150 user="$USER"

# Cluster-wide QoS catalog (what exists, with limits) — useful when someone
# references a QoS you haven't seen:
sacctmgr show qos format=name%25,maxwall,maxsubmit,maxtresperuser%40

# What's currently running/queued on your account, to estimate wait times:
squeue -A mrs_2 -o "%.10i %.12j %.8u %.10a %.12q %.6D %.10M %.20R"
```

Known QoS under account `mrs_2` at the time of writing (ordered worst-to-best
for interactive work):

| QoS | Preemptible? | Use case |
|---|---|---|
| `lowest` | yes | Scavenger. OK for 100% reproducible, checkpoint-heavy jobs only. |
| `h200_dev` | no | Quick interactive debug / smokes. **Use this for sprint-d1 and Phase A rho-1b.** |
| `h200_mrs_shared` | no | Shared MRS-pillar pool; used when team allocation is full. |
| `h200_mrs_2_high` | no | Team's dedicated allocation. Use for long / parallel production Phase A and Phase B runs. |

If `sacctmgr` shows you access to other accounts (e.g. a general `default`
or a different team account), the same `--account=<x> --qos=<y>` pattern
applies — just substitute.

#### 0.1.2 Reserve a node and attach a shell

```bash
# Pick size by workload:
#   Phase A rho-1b smoke     → gpu:1, 4h   (this validates the seed fix)
#   Phase A single 7B policy → gpu:1, 6h
#   Phase A all 4 in parallel→ gpu:4, 24h  (CUDA_VISIBLE_DEVICES=N per shell)
#   Phase B1 training        → gpu:4, 24h
salloc --account=mrs_2 --qos=h200_dev --gres=gpu:1 --time=4:00:00 --mem=0

# salloc returns once the allocation is granted. Find the node id, then
# attach a shell on it:
squeue -u "$USER"                     # copy the JOBID
srun --jobid=<JOBID> --pty bash
# NOTE: `ssh <node>` fails with "Permission denied (publickey)" on this
# cluster — you MUST go through `srun --pty` or `sbatch`.
```

#### 0.1.3 Activate the env and sanity-check (every session)

```bash
# Inside the compute node:
nvidia-smi                            # must list ≥1 H200/H100/A100
ldconfig -p | grep libcuda.so.1       # must print at least one hit

source /home/$USER/miniconda3/etc/profile.d/conda.sh
conda activate grpocredit
cd ~/projects/grpocredit

bash scripts/sanity_check_server2.sh  # fails fast on env / test / cache
# Expect: "[sanity] PASS — server2 looks ready." and pytest reports ≥ 103 passed.
# The count grows as hardening lands — below the low-water mark means your
# checkout is missing fixes: 42 (pre–seed-contract) → 66 (§2.3) → ~91 (§2.4
# verifier fix) → 103 (§2.5 infra-vs-policy split incl. detector-health
# reclassification). If you see ≤ 99, `git pull`.
```

Only after `sanity_check_server2.sh` exits 0 is it safe to run §2.

#### 0.1.4 Detach / re-attach without losing the allocation

`srun --pty bash` gives you an interactive shell on the reserved node. If
your ssh session drops, the SLURM allocation lives on (until `--time`
expires), but the `srun` shell dies with your terminal. Two options:

1. **Short jobs (Phase A rho-1b, ≤ 30 min).** Just re-run `srun --jobid=<JOBID>
   --pty bash` to re-attach — output of the running `run_oracle.sh` is
   captured in `experiments/oracle/.../launch.log` and wandb.
2. **Long jobs (Phase A 7B, Phase B).** Wrap the launcher in `tmux` or
   `screen` *before* starting it, or submit with `sbatch` instead of
   `salloc + srun`. A one-shot sbatch template:

   ```bash
   sbatch --account=mrs_2 --qos=h200_mrs_2_high --gres=gpu:1 --time=6:00:00 \
       --mem=0 --job-name oracle-rho1b \
       --output=experiments/oracle/rho1b_sft_gsm8k/slurm-%j.log \
       --wrap='source ~/miniconda3/etc/profile.d/conda.sh && \
               conda activate grpocredit && \
               cd ~/projects/grpocredit && \
               bash scripts/run_oracle.sh configs/oracle/rho1b_sft_gsm8k.yaml'
   ```

   Then `tail -f experiments/oracle/rho1b_sft_gsm8k/slurm-<JOBID>.log`.

## 1. One-time setup

### 1.1 Clone repos

```bash
# pick whatever workspace path is convention on server2:
cd ~/work && git clone <your-fork-of-GRPO_mcts.git> && cd GRPO_mcts/grpocredit
```

### 1.2 Recreate the `grpocredit` conda env

The env was built on the dev machine with the package set captured in
`grpocredit-server2-pip-freeze.txt` (173 packages, generated by `pip freeze
--exclude-editable`). Reproduce it on server2:

```bash
# 1) create the env shell
conda create -n grpocredit python=3.11 -y
conda activate grpocredit
conda env config vars set CUDA_HOME=$CONDA_PREFIX
conda deactivate && conda activate grpocredit  # re-activate to pick up CUDA_HOME

# 2) install torch first (must come before vllm — vllm pins torch 2.5.x)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# 3) install vllm (pulls transformers / sentence-transformers / etc.)
pip install vllm==0.6.4.post1

# 4) install flash-attn — prebuilt wheel for cu12 / torch 2.5 / py3.11
pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

# 5) install grpocredit + dev extras (editable)
pip install -e ".[dev]"

# 5b) make scripts/ importable as a package, regardless of cwd or how the
#     script is launched. Several scripts in scripts/ do `from scripts._shared
#     import ...`, which only resolves if the repo root is on sys.path; pip
#     install -e . only adds src/grpocredit, not the sibling scripts/ tree.
#
#     Caveat: gguf 0.10.0 (a transitive dep of vLLM's GGUF model loader) ships
#     its CLI helpers as a TOP-LEVEL package called `scripts/` in
#     site-packages, which masks our own. Because regular packages always beat
#     namespace-package portions during import, the .pth alone isn't enough —
#     we also have to delete the gguf shadow. We don't use any of gguf's CLIs
#     (gguf-dump etc.); the gguf Python module itself is untouched.
#
#     sanity_check_server2.sh does both of these idempotently and is the
#     recommended way to set this up; the manual equivalent is:
SITE=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
[ -f "$SITE/scripts/__init__.py" ] && grep -q gguf "$SITE/scripts/__init__.py" \
    && rm -rf "$SITE/scripts"   # remove gguf's bogus top-level scripts/ pkg
echo "$(pwd)" > "$SITE/grpocredit_repo.pth"

# 6) optional — pin everything else from the frozen list (recommended for
#    bit-for-bit reproducibility):
pip install --upgrade -r grpocredit-server2-pip-freeze.txt
```

Sanity check:

```bash
python -c "import torch, vllm, sentence_transformers, math_verify, grpocredit; \
print('torch', torch.__version__, 'cuda?', torch.cuda.is_available()); \
print('vllm', vllm.__version__); print('grpocredit OK')"
pytest -q  # expect 42 passed
```

> **NCCL overlay caveat.** On the dev machine we hit a subtle bug where
> `pip show nvidia-nccl-cu12` reported the right version (matching torch) but
> the on-disk `libnccl.so.2` had been overlaid with a CUDA-13 build, refusing
> to load on a CUDA-12 driver. Symptom: `Cuda failure 'CUDA driver version is
> insufficient for CUDA runtime version'` at the first NCCL collective. Fix:
> `pip install --force-reinstall --no-deps nvidia-nccl-cu12==<the version pip
> already records>`. If you hit this on server2, that's the one-liner.

### 1.3 Pre-download the 4 oracle policies

Doing this once before launching avoids a download race when the runs all
hit Hugging Face simultaneously, and lets you launch in air-gapped mode.

| Checkpoint | ~Size | Role |
|---|---|---|
| `realtreetune/deepseekmath-7b-sft-MATH-v2` | 14 GB | **Headline RL init** (MATH) |
| `realtreetune/deepseekmath-7b-sft-GSM8K` | 14 GB | **Headline RL init** (GSM8K) |
| `realtreetune/rho-1b-sft-GSM8K` | 3 GB | Cheap fast-iteration RL init |
| `Qwen/Qwen2.5-Math-7B-Instruct` | 14 GB | **Saturation-ceiling probe only** — NOT an RL init |

```bash
huggingface-cli download realtreetune/deepseekmath-7b-sft-MATH-v2
huggingface-cli download realtreetune/deepseekmath-7b-sft-GSM8K
huggingface-cli download realtreetune/rho-1b-sft-GSM8K
huggingface-cli download Qwen/Qwen2.5-Math-7B-Instruct
```

### 1.4 wandb

```bash
wandb login         # interactive; or set WANDB_API_KEY in env
export WANDB_PROJECT=grpo-voi
```

## 2. Phase A — launch the oracle on 4 policies

```bash
cd ~/work/GRPO_mcts/grpocredit

# Cheapest / fastest first — flushes any infra bugs on the 1.4B model.
bash scripts/run_oracle.sh configs/oracle/rho1b_sft_gsm8k.yaml

# Headline RL inits (each ~ a few GPU-hours depending on rollout cost).
bash scripts/run_oracle.sh configs/oracle/deepseek_math_sft.yaml          # MATH-v2
bash scripts/run_oracle.sh configs/oracle/deepseek_math_sft_gsm8k.yaml    # GSM8K

# Saturation-ceiling probe — Qwen-Instruct is NOT an RL init; this run
# produces the §5 informative-group-fraction figure that motivates VoI.
bash scripts/run_oracle.sh configs/oracle/qwen_math_instruct.yaml

# Or all four sequentially:
for c in configs/oracle/*.yaml; do bash scripts/run_oracle.sh "$c"; done
```

Each invocation runs Day 1 → Day 2A → Day 2B → Day 3 in order and exits
non-zero if any gate fails. The Day 1 step now includes the
`sft_warmup_plan.md §5` group-variance probe (256 prompts × G=8 → fraction
of informative groups), and Day 3 reads its output and adds it as the first
row of the gate decision table. **Exit code 6 = group-variance gate failed
= wrong starting policy, do not RL on this `π_ref`.**

**Per-policy outputs land at:**

```
experiments/oracle/{rho1b_sft_gsm8k,deepseekmath_sft_math_v2,deepseekmath_sft_gsm8k,qwen_math_instruct_saturation_ceiling}/
├── day1_rollouts.jsonl              # smoke test trajectories
├── day1_boundaries.json             # 5-15 boundaries/traj target
├── day1_verifier_accuracy.txt       # ≥ 0.95 expected on the 7B SFT'd policies
├── day1_group_variance.json         # NEW — sft_warmup_plan.md §5 gate
├── concordance_mi.json              # Day 2A
├── concordance_per_position.csv
├── oracle_q_variance.json           # Day 2B
├── oracle_kappa.txt                 # κ scalar with bootstrap CI
├── oracle_position_curve.csv        # decile curve — drives Stage-1 w_pos_shape decision
├── oracle_correlations.json         # ρ for H_token / H_sem / s_2 with Fisher-z CIs
├── GATE_REPORT.md                   # Day 3 decision table — read this first
├── gate_decision.json               # machine-readable gate verdict
└── launch.log                       # provenance: timestamp / config / params
```

**`run_oracle.sh` exit codes (from sprint_d3):**

| Code | Meaning |
|---|---|
| 0 | proceed to main phase |
| 2 | proceed with caveats (one threshold marginal — usually ρ near gate) |
| 3 | pilot run required before commit |
| 4 | pivot to Plan B (concordance MI failed) |
| 5 | data missing — re-run failed step |
| **6** | **policy-class gate failed at step 0** — either ``boundaries_mean`` below threshold (π_ref produces short CoTs — expected on rho-1b at ~2.3/traj) OR ``fraction_informative`` below §5 threshold. This is a **policy-distribution signal**, not a code bug. Expected on Qwen-Instruct (saturation ceiling); borderline on rho-1b (short-CoT + bimodal GSM8K difficulty — see §2.5); a sign of a bad SFT recipe on the 7B SFT'd models. For an intentionally-weak debug policy, override with `PROCEED_ON_POLICY_GATE_FAIL=1 bash scripts/run_oracle.sh …` or `--proceed-on-policy-gate-fail` — see §2.5. |
| **7** | **rollout-diversity sentinel failed — backend bug, not a policy verdict.** See §2.3 below. Rerun after fixing the backend; do not interpret as a πref signal. |

### 2.3 Rollout-diversity sentinel (exit 7)

Day 1 now runs a tiny `G=4, n_prompts=8` probe (§`sprint_d1_infra_smoke
.py` before the group-variance probe) and asserts via
`grpocredit.oracle.rollout_diversity.assert_diverse_rollouts` that at
temperature 0.9 the runner actually produces distinct response texts per
group. A violation (average unique-fraction < 0.5 or more than 50 % of
groups fully collapsed) trips exit code 7.

Why it exists: `vLLM 0.6.4.post1 + enable_prefix_caching=True + a fixed
per-request SamplingParams.seed + n > 1 + instruct-style shared prompt
prefixes` is a known collapse mode — it was the mechanism behind the
original `rho-1b-sft-GSM8K` failure (`mean_group_reward_std=0.0` across
256/256 groups). The §5 group-variance gate can't distinguish that
"backend is producing 8 copies of the same completion" from "the policy
itself is saturated", because both look like informative_fraction=0. The
sentinel disambiguates by looking at raw response text collisions before
the verifier runs.

The runner enforces the fix at the API boundary (`vllm_runner.py`'s
`_sampling_params`): per-request `seed` is silently dropped for `n > 1`
rollout calls, matching verl / TRL / OpenRLHF convention. For paper-
figure bit-reproducibility, set `rollout.deterministic_n: true` to enable
the VinePPO `num_expansion_rounds` pattern (single `n=N` request is fanned
out into N `n=1` requests with seeds `[base, base+1, …]`). Engine-level
reproducibility (`LLM(..., seed=cfg.rollout.seed)`) is unchanged.

> ⚠ **Do not set `rollout.deterministic_n: true` globally** unless you
> specifically want bit-reproducible oracle tables for the paper. It loses
> throughput — every `n > 1` call becomes `n` serial single-sample calls,
> which with `enable_prefix_caching=True` is cheap-ish but still ~N× decode
> passes for no scientific gain on the smoke test or the gates. The default
> (drop-seed) matches verl / TRL / OpenRLHF convention and is what every
> step of this runbook (`sprint_d1`, §5 group-variance gate, Day 2 oracle
> forced-action rollouts, Day 2 concordance MI) needs. Turn it on only when
> generating the final `GATE_REPORT.md` numbers you intend to publish, and
> even then scope it to the one config you're freezing via a tiny overlay
> YAML rather than editing the base:
>
> ```yaml
> # configs/oracle/deepseek_math_sft_gsm8k_paperfreeze.yaml
> extends: deepseek_math_sft_gsm8k.yaml
> name: oracle_deepseek_math_sft_gsm8k_paperfreeze
> output_dir: experiments/oracle/deepseek_math_sft_gsm8k_paperfreeze
> rollout:
>   deterministic_n: true
> ```
>
> ```bash
> bash scripts/run_oracle.sh configs/oracle/deepseek_math_sft_gsm8k_paperfreeze.yaml
> ```
>
> Leave all other configs (including `base_qwen_math.yaml`,
> `base_deepseekmath_sft.yaml`, and the four per-policy oracle configs) at
> the default `deterministic_n: false`.

### 2.4 Verifier extraction priority (silent grader bug)

Even after the seed fix (§2.3) and the prompt-template fix (§2.2), a
**third** class of failure can crush the §5 gate while the model is
actually performing on-spec: the verifier picks up an intermediate
arithmetic step as the "final answer" and marks correct rollouts wrong.
Concretely, the regression that was caught on `rho-1b-sft-GSM8K`:

* Model emitted: `"Natalia sold 48/2 = 24 clips in May. Natalia sold 48 + 24 = 72 clips altogether in April and May.\n#### 72"`
* ground-truth: `"72"`
* Pre-fix verifier extracted: `"24 clips in May"` (from the first `= 24` in the CoT).
* Pre-fix verifier graded: **wrong**.

Aggregate impact measured on 100 GSM8K rollouts: true pass@1 ≈ 70 %,
reported pass@1 = 15 %. The §5 gate's `fraction_informative=0.121`
signal was almost entirely spurious.

The fix in `src/grpocredit/rollout/verifier.py` is a priority-ordered
registry of extractors in `_EXTRACTORS`. Each entry is `(method_tag,
extractor_fn)` where `extractor_fn(response) -> str` returns `""` to
yield to the next extractor. Current registry:

| Priority | Method tag | Convention | Used by |
|---|---|---|---|
| 1 | `gsm8k_hash` | `#### X` at end | GSM8K-SFT'd models (rho-1b, deepseekmath-sft-GSM8K, VinePPO's) |
| 2 | `answer_tag` | `<answer>X</answer>` | DeepSeek-R1 and R1-distilled models |
| 3 | `boxed` | last `\boxed{X}` | Qwen-Math-Instruct, deepseekmath-sft-MATH, any MATH-trained model |
| 4 | `answer_is` | "(the )?(final )?answer (is\|:\|=) X" prose | most instruct-tuned models without a strict template |
| 5 | `fallback` | last numeric token | weak last-resort; method tag is the "don't trust me" signal |

What the pre-fix version got wrong: `answer_is` ran *before* `gsm8k_hash`
AND its regex also matched bare `= X`, so every intermediate CoT step
(`48/2 = 24 …`) beat the authoritative `#### 72`. Fix: reorder, and
narrow `_ANSWER_IS_RE` to `"(the )?(final )?answer (is|:|=) X"` phrasing
only. Frozen by `tests/test_verifier.py::
test_gsm8k_hash_beats_intermediate_equation_*` and
`test_answer_is_no_longer_matches_bare_equation`.

**Coverage.** GSM8K, MATH, MATH-500, AIME-24, OlympiadBench (free-form
numeric/closed-form). Any rho-1b / deepseekmath / Qwen-Math / R1 /
R1-distilled policy should grade correctly against any of those datasets
without code changes.

**Diagnostic when you suspect grader drift.** Run
`scripts/inspect_day1_rollouts.py --tokenizer <repo>` and inspect the
`[aggregate] verifier extract method:` line. If it shows anything other
than `gsm8k_hash` dominant (for GSM8K configs) or `boxed` dominant (for
MATH configs), you have a grader-mismatch, not a model-quality issue.

#### Extending the verifier for a new model/dataset

**When a new model/dataset needs a new extractor** — add it in 4 steps:

```python
# 1. Write the extractor (src/grpocredit/rollout/verifier.py).
def _extract_mynewformat(response: str) -> str:
    m = re.search(r"FINAL: (\d+)", response)
    return m.group(1) if m else ""

# 2. Insert into the registry at the right priority.
#    Higher priority = checked first = more authoritative.
_EXTRACTORS.insert(3, ("mynewformat", _extract_mynewformat))

# 3. Add a test in tests/test_verifier.py with a verbatim response.
# 4. Update the table above and the `VerifierResult.method` docstring.
```

**When a new benchmark doesn't fit this shape at all** (multiple-choice
letters, code execution, Lean proofs) — don't bolt it onto this
registry. Write a new `XXXVerifier` class with the same `score(response,
ground_truth) -> VerifierResult` contract and wire it into the oracle
config. The oracle pipeline is verifier-agnostic.

**What is *not* covered** and would need a separate verifier class:

| Benchmark shape | Why this registry is wrong for it | Fix |
|---|---|---|
| Multiple-choice (MMLU, ARC, GPQA) | Answer is a letter ("A"/"B"/"C"/"D"); our numeric fallback misfires on explanation digits | New `MultipleChoiceVerifier` |
| Code generation (HumanEval, MBPP) | Answer is executable code; needs sandbox exec to check test-case pass rate | `sandbox` / `bigcode_eval`-style verifier |
| Lean / Coq proofs | Answer is a proof script; needs a proof checker | Proof-checker-based verifier |
| Free-form string answers (open-QA, summarisation) | No canonical normalisation; needs semantic-similarity scoring | LLM-as-judge or embedding-based verifier |

### 2.5 Policy gate vs. infra gate — and `--proceed-on-policy-gate-fail`

After §2.3 and §2.4 landed, the Day-1 smoke test separates two fundamentally
different failure classes that used to share `exit 1`:

| Class | Triggers | What it means | Default behaviour |
|---|---|---|---|
| **Infra** | `boundaries_max == 0` (detector segmented nothing anywhere) OR `verifier_accuracy < 0.9` | Bug in our code/config — detector broken, or grader mis-extracting. These are the only two signals that pin "your code is wrong". | `exit 1` — blocks Day 2 unconditionally. **Not** overrideable. |
| **Policy** | `boundaries_mean < 3` (short CoTs — a distribution property; rho-1b on GSM8K runs ~2.3) OR `fraction_informative < pass_threshold` (§5 gate, default 0.5) | Quality signal about `π_ref` — model's typical trajectory is shorter than MATH-7B-style reasoning, and/or most groups are saturated. **Not** a bug; the infra is producing honest numbers. | `exit 6` — blocks Day 2 by default. Overrideable per run. |

**Why `boundaries_mean` moved from infra to policy.** Originally `mean < 3`
was classified as infra ("detector broken"), but that conflated detector
health with trajectory length. A 1 B concise-solver on GSM8K produces
2-3-step solutions by design and will average ~2.3 boundaries/traj; that
is not a broken detector, it is the model's trained output distribution.
The honest infra signal is `boundaries_max == 0` — "the detector produced
*zero* boundaries on *every* trajectory" — which can only be a code bug.
If the detector works on *any* trajectory (our observed `boundaries_max=14`
on rho-1b), the mean is a policy property, not a code-health property.
`grpocredit.oracle.stop_gate.classify_stop_gate` now enforces this split.

**When to override the policy gate.** Only for π_ref that are expected to
flunk policy-class thresholds by design:

* `rho-1b-sft-GSM8K` on GSM8K — plan §3.A debug policy, not a headline
  RL init. Short CoTs + bimodal difficulty → both policy signals trip.
  See §2.5.1 row 1 for the numerical decoding.
* `Qwen2.5-Math-7B-Instruct` on GSM8K — saturation ceiling on purpose
  (§2.1 expected pattern, group-variance ≈ 0.30–0.40). Paper figure,
  not something we RL on.

For both, the Day-2/3 oracle numbers (`κ`, `ρ`, `MI`) are still
meaningful — they're computed over the `fraction_informative` slice
of groups that do have signal.

**How to override.** Two equivalent paths — both keep the failure loudly
recorded in `day1_summary.json` and the wandb run summary, and both still
respect infra failures:

```bash
# Env-var (recommended — works transparently with run_oracle.sh):
PROCEED_ON_POLICY_GATE_FAIL=1 bash scripts/run_oracle.sh \
    configs/oracle/rho1b_sft_gsm8k.yaml

# Direct CLI flag (useful when running sprint_d1 standalone):
python scripts/sprint_d1_infra_smoke.py \
    --config configs/oracle/rho1b_sft_gsm8k.yaml \
    --output-dir experiments/oracle/rho1b_sft_gsm8k \
    --proceed-on-policy-gate-fail
```

Do NOT set this globally in a shell rc. Use it per-invocation for policies
that you've reasoned-through as intentionally-gate-borderline (rho-1b,
Qwen-Instruct). For DeepSeekMath-SFT and any own-SFT'd model, gate failure
means the SFT recipe is wrong — fix the recipe, do not override.

The `GATE_REPORT.md` emitted at Day 3 (`sprint_d3_gate_report.py`) reads
`day1_group_variance.json` directly and issues its own `GROUP-VAR: fail`
/ `proceed-with-caveats` verdict against the same 0.5 threshold. That
means the final decision table stays honest — overriding the Day-1 stop-
gate only lets Day 2 compute the oracle numbers; Day 3 still reports the
gate failure in `GATE_REPORT.md` and `gate_decision.json` and will issue
exit code 2 / 3 / 4 accordingly. If you want to see the infra-vs-policy
split at a glance, inspect `day1_summary.json`'s new `infra_fail`,
`policy_fail`, and `proceed_on_policy_gate_fail` fields alongside
`GATE_REPORT.md`.

#### 2.5.1 Reference decoding table — how to read a Day-1 FAIL

The first time a Day-1 run fails on a new policy/dataset combo, classify
each triggered signal against this table before touching anything. The
rightmost column tells you *whether to code* vs *whether to override*.
Do not skip this step — the fact that `boundaries_mean < 3` spent a
whole week looking like an infra bug on rho-1b is exactly what this
table prevents from happening again.

This is the decoding of the exact rho-1b-sft-GSM8K GSM8K-test run that
motivated the reclassification — keep it here as the concrete reference:

| # | Check (as originally posed) | Observed | Still triggering? | Real nature | What to do |
|---|---|---|---|---|---|
| 1 | Infra — `boundaries_mean < 3` | **2.26** | Yes | **Miscategorised.** A policy-distribution property (rho-1b writes short GSM8K solutions — 2-3 steps by design). `boundaries_max = 14` *proves* the detector works, so this is not an infra bug. | Reclassify to POLICY (done — `stop_gate.classify_stop_gate`). Override with `--proceed-on-policy-gate-fail` for rho-1b; for larger models on MATH this threshold is still a useful floor. |
| 2 | Infra — `verifier_accuracy < 0.9` | 1.0 | No | Clean. Verifier §2.4 fixes landed and the grader handles `rho-1b`'s `#### X` format correctly. | Nothing. |
| 3 | Policy — `fraction_informative < 0.5` | 0.492 | Yes (borderline) | Real policy signal — bimodal GSM8K difficulty: ~19 % always-solved + ~32 % always-failed + ~49 % informative. 0.25 σ below the 0.5 threshold at the `G=8, N=256` measurement-noise floor; another run could flip either side. | Override for rho-1b (debug policy, plan §3.A). Don't override for DeepSeekMath-SFT or own-SFT'd headlines — there, 0.49 would mean the SFT recipe is wrong. |

**How to classify each trigger yourself**, for any *future* Day-1 failure:

1. Read `day1_summary.json` — the new `infra_fail`, `policy_fail`,
   and `stop_gate_reasons` fields give you the classification without
   guessing. `stop_gate_reasons` lists each triggered sub-check with
   numeric evidence.
2. If `infra_fail: True`, do not override. Fix the code. The only two
   paths to `infra_fail` are `boundaries_max == 0` (detector completely
   silent — the detector code is broken, check
   `grpocredit.voi.boundary_detect`) and `verifier_accuracy < 0.9`
   (grader broken — re-run `tests/test_verifier.py` and inspect
   `scripts/inspect_day1_rollouts.py`).
3. If only `policy_fail: True`, decide whether the policy is in the
   explicit "weak debug policy" list (rho-1b, Qwen-Instruct) and override
   per §2.5. For any other policy, a policy-class fail means **the SFT
   recipe is wrong**, not that the gate is too strict; go fix the
   recipe.

### 2.1 Cross-policy comparison

After all four runs finish, the headline picture comes from the gate reports.
Assemble a comparison table (one row per policy) with columns: **group-
variance fraction** (the §5 gate value), `κ`, `ρ_gate`, `ρ(H_token, Var Q^π)`,
`ρ(H_sem, Var Q^π)`, `ρ(s_2, Var Q^π)`, `MI(C_prefix; C_terminal)`, and
Day-3 verdict.

Expected pattern (the figure that motivates the paper):

* **Group-variance fraction:** Qwen-Instruct ≈ 0.30–0.40 (fails the §5
  gate, exit 6 — that's the saturation-ceiling story), DeepSeek SFT'd ≈
  0.85–0.95 (passes), rho-1b ≈ 0.60–0.85 (passes more weakly because the
  smaller model has lower pass@1 floor). This **delta itself** is a paper
  figure: "fraction of GRPO groups with informative advantage at step 0
  across starting policies".
* **κ:** should **increase** with stronger SFT'd models (more room for VoI
  selection to matter), so DeepSeek ≥ rho-1b. Qwen-Instruct's κ is
  uninterpretable when the §5 gate fails — most of its variance signal
  comes from the few non-saturated groups, which is a heavily filtered
  subpopulation.
* **`ρ`** correlations on entropy-based signals tend to **decrease** as
  models sharpen — bad news for `H_token` as a signal, neutral for `H_sem`.
* **`MI`** should be higher at mid-trajectory boundaries than at very-early
  ones across all four — that's the position curve (§2B).

The headline RL init for the main paper is **DeepSeekMath SFT'd** (Option B
of `sft_warmup_plan.md` §4.1). Qwen-Instruct's role is the saturation
ceiling, not a primary-policy run. If `κ < 2` on DeepSeek SFT'd, the paper
pivots to efficiency-only per `experiment_plan_grpo_voi.md` §0 D2.

### 2.2 Known caveats

* ~~**rho-1b `boundaries_mean` classified as infra.**~~ **RESOLVED.**
  `boundaries_mean < 3` used to trigger `infra_fail=True` and exit 1 for
  `rho-1b-sft-GSM8K` on GSM8K (observed 2.26 with `max=14` — detector
  works fine, model just writes short CoTs). Reclassified to POLICY in
  `grpocredit.oracle.stop_gate.classify_stop_gate`. The infra-class
  boundary check is now `boundaries_max == 0` (detector silent across
  all trajectories), which cannot false-positive on legitimate short
  outputs. See §2.5 and §2.5.1 for the full decoding table.

* ~~**rho-1b prompt template.**~~ **RESOLVED.** We empirically hit this
  exactly as warned — after the seed fix (§2.3) cleared the diversity
  collapse, `rho-1b-sft-GSM8K` was still landing pass@1 ≈ 6 % under
  `math_instruct` + temp 0.9 + top_p 0.95 (vs the published ~35 %), which
  crushed the §5 gate to `fraction_informative=0.195`. The fix that lives
  in the repo now:

  * `grpocredit.rollout.datasets._apply_vineppo_math_task` — VinePPO's
    verbatim SFT-time template `"[MATH_TASK] Problem:\n{q}\n\nSolution:"`
    (lifted from VinePPO-grpo's
    `configs/prompt_library/generic_{GSM8K,MATH}_step_by_step.jsonnet`
    and `configs/sft_rho1b_for_gsm8k.jsonnet`).
  * `configs/oracle/rho1b_sft_gsm8k.yaml`,
    `configs/oracle/deepseek_math_sft.yaml` (MATH-v2),
    `configs/oracle/deepseek_math_sft_gsm8k.yaml` — all three switched
    to `prompt_template: vineppo_math_task` and VinePPO's eval-time
    sampling settings (`temperature: 0.35`, `top_p: 0.9`,
    `stop: ["\n\n\nProblem:"]`).
  * `configs/oracle/qwen_math_instruct.yaml` — intentionally unchanged
    (Qwen-Instruct IS chat-trained with a tokenizer-exposed template).
  * `tests/test_prompt_templates.py` pins the template string and the
    three oracle configs' sampling settings so this can't drift silently.

  If you are running from an older checkout and see pass@1 on `rho-1b`
  materially below ~30 %, `git pull` and re-launch — that was the bug.

* **GPU memory.** vLLM's `gpu_memory_utilization` defaults to 0.9 in
  `base_qwen_math.yaml`. On an 80 GB GPU that's ~72 GB for vLLM weights+KV
  cache, plenty for any of the 3 policies. On a 40 GB GPU you'll need to
  drop to 0.85 for the 7B configs — override on the launcher:
  `python -c "from grpocredit.common.config import load_config; cfg=load_config('configs/oracle/qwen_math_instruct.yaml', overrides={'rollout.gpu_memory_utilization': 0.85}); ..."`
  or just edit the YAML.

* **Wall-clock estimates** (single 80 GB H100 / A100):
  * rho-1b GSM8K: ~30 min total (Days 1+2A+2B+3)
  * deepseek-math-sft: ~3–4 h
  * Qwen-Math-Instruct: ~3–4 h

  Total Phase A: ~7–9 h on one GPU. Trivially parallelizable across 2–3 GPUs
  if you launch them in different shells.

## 3. Phase B1 — already shipped

See `VinePPO/SERVER2_RUNBOOK.md`. Trains VinePPO and a newly-ported GRPO
trainer inside treetune, both starting from the same VinePPO-published
SFT'd checkpoints (Option B). Output: paired wandb-logged accuracy / KL /
advantage-variance curves directly comparable.

## 4. Phase B2a — placeholder (verl GRPO baseline + VoI on DeepSeekMath SFT'd)

Headline experiment per `sft_warmup_plan.md` §4.1. RL starts from
`realtreetune/deepseekmath-7b-sft-{MATH-v2,GSM8K}` (matched to VinePPO's
own start). The `configs/base_deepseekmath_sft.yaml` added in this code
drop is the foundation. Code (verl fork + `grpo_voi_trainer.py` + launch
scripts) lands on the next "continue".

## 5. Phase B2b — placeholder (own-SFT'd Qwen-base generalization run)

Generalization study per `sft_warmup_plan.md` §3.A and §4.2. SFT
`Qwen2.5-Math-7B` (base, not Instruct) ourselves on GSM8K + MATH train
splits with the recipe in §3.A (lr 1e-5, 3 epochs, lr cosine + 3% warmup),
then RL with the same verl pipeline as B2a. Validates the credit-
assignment gain replicates on a modern base. Code lands after B2a.
