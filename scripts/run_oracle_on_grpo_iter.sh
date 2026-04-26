#!/usr/bin/env bash
# Phase A oracle on a GRPO **in-training** iter checkpoint.
#
# This wrapper runs the same 4-step pipeline as `scripts/run_oracle.sh`
# (Day 1 → Day 2A → Day 2B → Day 3) but points the oracle at a checkpoint
# emitted by VinePPO/treetune's GRPO run, instead of an HF-Hub starting policy.
#
# Workflow on server2:
#   1. Launch GRPO in VinePPO/ — see VinePPO/scripts/launch_server2_grpo.sh.
#      It writes per-iter HF dumps to:
#        VinePPO/experiments/grpo_<model>_<task>/checkpoints/ckpt--iter_NNNN--*/hf_pretrained/
#   2. After each iter you care about, run this script to oracle that
#      checkpoint (or run all of them at the end of training, on the same GPU
#      or a second one — they're independent).
#
# Usage:
#   bash scripts/run_oracle_on_grpo_iter.sh <model> <task> <iter> [base_oracle_yaml]
#
#   model ∈ {rho1bSft2, deepseekSft2}
#   task  ∈ {GSM8K, MATH}
#   iter  integer (0 = end of GRPO iter 0, i.e. one round of policy updates)
#   base_oracle_yaml  (optional) overrides the auto-picked base oracle config
#
# Examples:
#   bash scripts/run_oracle_on_grpo_iter.sh rho1bSft2  GSM8K 0
#   bash scripts/run_oracle_on_grpo_iter.sh rho1bSft2  GSM8K 1
#   bash scripts/run_oracle_on_grpo_iter.sh deepseekSft2 GSM8K 2
#
# Env-vars:
#   VINEPPO_DIR     path to VinePPO repo (default: ../VinePPO relative to this repo)
#   GRPO_RUN_DIR    full path to a GRPO experiments dir; overrides VINEPPO_DIR
#                   resolution (use this if you set APP_DIRECTORY at launch time)
#   N_TRAJ          override oracle.n_trajectories (default: read from base YAML)
#   PROBE           verifier-probe size for Day 1 (default: 200)
#   CKPT_SUBDIR     subdir under ckpt--iter_NNNN--* that holds the HF dump
#                   (default: hf_pretrained — VinePPO trainer convention)
#
# Output:
#   experiments/oracle/grpo_iter/<model>_<task>/iter_<NNNN>/
#       day1_*.json                     — Day 1 smoke + group-variance probe
#       concordance_*.{json,csv}        — Day 2A
#       oracle_*.{json,txt,csv}         — Day 2B (κ, ρ, position curve)
#       GATE_REPORT.md                  — Day 3 decision table
#       gate_decision.json
#       launch.log
#
# Exit code: forwarded from sprint_d3_gate_report.py
#   0 proceed · 2 caveats · 3 pilot · 4 Plan B · 5 missing · 6 group-variance fail

set -euo pipefail

if [ $# -lt 3 ]; then
    echo "usage: $0 <model: rho1bSft2|deepseekSft2> <task: GSM8K|MATH> <iter:int> [base_oracle_yaml]" >&2
    exit 64
fi

MODEL="$1"
TASK="$2"
ITER="$3"
BASE_YAML_OVERRIDE="${4:-}"

case "$MODEL" in rho1bSft2|deepseekSft2) ;; *) echo "bad model: $MODEL" >&2; exit 64;; esac
case "$TASK" in GSM8K|MATH) ;; *) echo "bad task: $TASK" >&2; exit 64;; esac

if ! [[ "$ITER" =~ ^[0-9]+$ ]]; then
    echo "iter must be a non-negative integer, got: $ITER" >&2
    exit 64
fi

cd "$(dirname "$0")/.."
GRPOCREDIT_ROOT="$(pwd)"

# 1. Pick the base oracle YAML + the original SFT'd HF-Hub repo (for tokenizer
#    pinning — VinePPO's `_save_hf_pretrained` saves model weights only, not
#    the tokenizer, so we have to point the tokenizer at the original repo).
if [ -z "$BASE_YAML_OVERRIDE" ]; then
    case "${MODEL}_${TASK}" in
        rho1bSft2_GSM8K)
            BASE_YAML="configs/oracle/rho1b_sft_gsm8k.yaml"
            TOKENIZER_REPO="realtreetune/rho-1b-sft-GSM8K"
            ;;
        deepseekSft2_GSM8K)
            BASE_YAML="configs/oracle/deepseek_math_sft_gsm8k.yaml"
            TOKENIZER_REPO="realtreetune/deepseekmath-7b-sft-GSM8K"
            ;;
        deepseekSft2_MATH)
            BASE_YAML="configs/oracle/deepseek_math_sft.yaml"
            TOKENIZER_REPO="realtreetune/deepseekmath-7b-sft-MATH-v2"
            ;;
        rho1bSft2_MATH)
            echo "rho-1b-sft-MATH is not a published checkpoint (only rho-1b-sft-GSM8K)." >&2
            echo "Pass an explicit base_oracle_yaml as the 4th arg if you have a custom config." >&2
            exit 64
            ;;
        *) echo "no base oracle yaml mapped for ${MODEL}_${TASK}" >&2; exit 64;;
    esac
else
    BASE_YAML="$BASE_YAML_OVERRIDE"
    # When the user supplies a custom base, they're responsible for the
    # tokenizer mapping — read it from the env or fall back to the override
    # YAML's own model.name_or_path.
    TOKENIZER_REPO="${TOKENIZER_REPO:-}"
fi

[ -f "$BASE_YAML" ] || { echo "base oracle yaml not found: $BASE_YAML" >&2; exit 1; }

# 2. Resolve the GRPO run dir.
if [ -n "${GRPO_RUN_DIR:-}" ]; then
    RUN_DIR="$GRPO_RUN_DIR"
else
    VINEPPO_DIR="${VINEPPO_DIR:-$GRPOCREDIT_ROOT/../VinePPO}"
    RUN_DIR="$VINEPPO_DIR/experiments/grpo_${MODEL}_${TASK}"
fi
[ -d "$RUN_DIR" ] || { echo "GRPO run dir not found: $RUN_DIR" >&2; exit 1; }

# 3. Resolve the iter checkpoint (glob — there's exactly one ckpt per iter).
CKPT_SUBDIR="${CKPT_SUBDIR:-hf_pretrained}"
ITER_PADDED="$(printf '%04d' "$ITER")"
GLOB="$RUN_DIR/checkpoints/ckpt--iter_${ITER_PADDED}--*"
# shellcheck disable=SC2206
matches=( $GLOB )
if [ ${#matches[@]} -eq 0 ] || [ ! -e "${matches[0]}" ]; then
    echo "no checkpoint matching $GLOB" >&2
    echo "available iters under $RUN_DIR/checkpoints/:" >&2
    ls -1d "$RUN_DIR/checkpoints/ckpt--iter_"* 2>/dev/null | sed 's/^/  /' >&2 || echo "  (none)" >&2
    exit 1
fi
if [ ${#matches[@]} -gt 1 ]; then
    echo "multiple checkpoints match $GLOB (expected exactly one):" >&2
    printf '  %s\n' "${matches[@]}" >&2
    exit 1
fi
CKPT_DIR="${matches[0]}"
HF_DIR="$CKPT_DIR/$CKPT_SUBDIR"
[ -d "$HF_DIR" ] || { echo "HF dump not found at $HF_DIR (set CKPT_SUBDIR if your trainer uses a different name)" >&2; exit 1; }
HF_DIR_ABS="$(cd "$HF_DIR" && pwd)"

# 4. Output dir for this iter's oracle.
OUT="experiments/oracle/grpo_iter/${MODEL}_${TASK}/iter_${ITER_PADDED}"
mkdir -p "$OUT"

# 5. Build a temp overlay YAML that extends the base and overrides
#    model.name_or_path + output_dir + wandb metadata.
OVERLAY="$(mktemp -t oracle_grpo_iter_XXXXXX.yaml)"
trap 'rm -f "$OVERLAY"' EXIT

BASE_YAML_ABS="$(cd "$(dirname "$BASE_YAML")" && pwd)/$(basename "$BASE_YAML")"
{
    echo "# Auto-generated by run_oracle_on_grpo_iter.sh — DO NOT EDIT."
    echo "extends: ${BASE_YAML_ABS}"
    echo
    echo "name: oracle_grpo_iter_${MODEL}_${TASK}_iter${ITER_PADDED}"
    echo "output_dir: ${OUT}"
    echo
    echo "model:"
    echo "  name_or_path: ${HF_DIR_ABS}"
    if [ -n "$TOKENIZER_REPO" ]; then
        echo "  tokenizer_name_or_path: ${TOKENIZER_REPO}"
    fi
    echo "  dtype: bfloat16"
    echo "  trust_remote_code: true"
    echo
    echo "wandb:"
    echo "  tags: [phase_a_iter, oracle, grpo_iter, ${MODEL}, ${TASK}]"
    echo "  group: grpo_iter_${MODEL}_${TASK}"
    echo "  job_type: oracle_iter"
} > "$OVERLAY"

echo "[oracle-iter] model=$MODEL task=$TASK iter=$ITER"
echo "[oracle-iter] base yaml: $BASE_YAML"
echo "[oracle-iter] checkpoint: $HF_DIR_ABS"
echo "[oracle-iter] overlay yaml: $OVERLAY"
echo "[oracle-iter] output dir: $OUT"

# Provenance
{
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) | base=$BASE_YAML | ckpt=$HF_DIR_ABS | iter=$ITER"
} >> "$OUT/launch.log"

# 6. Hand off to the standard pipeline.
N_TRAJ_ARG=()
if [ -n "${N_TRAJ:-}" ]; then N_TRAJ_ARG=("$N_TRAJ"); fi
PROBE_ARG=()
if [ -n "${PROBE:-}" ]; then PROBE_ARG=("$PROBE"); fi

bash scripts/run_oracle.sh "$OVERLAY" "${N_TRAJ_ARG[@]}" "${PROBE_ARG[@]}"
