#!/usr/bin/env bash
# Pre-flight sanity check for server2 before launching VinePPO / GRPO / oracle.
# Run this once after cloning + creating both conda envs. Exits non-zero on any
# failure so it can be wired into a bigger launch script.
#
# Usage:
#   bash grpocredit/scripts/sanity_check_server2.sh
#
# Assumes:
#   - $GRPO_MCTS points at the GRPO_mcts repo root, OR this script is run from
#     somewhere under that tree.
#   - Both `vineppo` and `grpocredit` conda envs already exist (per the two
#     SERVER2_RUNBOOK.md files).

set -euo pipefail

# Resolve repo root.
if [ -n "${GRPO_MCTS:-}" ] && [ -d "$GRPO_MCTS" ]; then
    ROOT="$GRPO_MCTS"
else
    HERE="$(cd "$(dirname "$0")" && pwd)"
    # this script lives at grpocredit/scripts/sanity_check_server2.sh
    ROOT="$(cd "$HERE/../.." && pwd)"
fi
[ -d "$ROOT/VinePPO" ] && [ -d "$ROOT/grpocredit" ] || {
    echo "FAIL: could not locate VinePPO/ and grpocredit/ under $ROOT" >&2
    exit 1
}
echo "[sanity] repo root: $ROOT"

# 0. Driver / GPUs.
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "FAIL: nvidia-smi not on PATH" >&2; exit 1
fi
N_GPU="$(nvidia-smi -L | wc -l)"
echo "[sanity] GPUs visible: $N_GPU"
[ "$N_GPU" -ge 1 ] || { echo "FAIL: zero GPUs visible" >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# 1. conda envs exist.
source "$(conda info --base)/etc/profile.d/conda.sh"
for env in vineppo grpocredit; do
    if ! conda env list | awk '{print $1}' | grep -qx "$env"; then
        echo "FAIL: conda env '$env' missing — see SERVER2_RUNBOOK.md to recreate" >&2
        exit 1
    fi
done
echo "[sanity] conda envs OK"

# 2. vineppo env: torch / vllm / deepspeed import + treetune unit tests.
echo "[sanity] === vineppo env ==="
conda activate vineppo
python -c "
import torch, vllm, deepspeed, transformers
print('  torch', torch.__version__, 'cuda?', torch.cuda.is_available())
print('  vllm', vllm.__version__)
print('  deepspeed', deepspeed.__version__)
print('  transformers', transformers.__version__)
"
( cd "$ROOT/VinePPO" && PYTHONPATH=src pytest tests/test_grpo_advantage.py -q )
conda deactivate

# 3. grpocredit env: imports + full unit-test suite + load_config smoke.
echo "[sanity] === grpocredit env ==="
conda activate grpocredit
python -c "
import torch, vllm, sentence_transformers, math_verify, grpocredit
print('  torch', torch.__version__, 'cuda?', torch.cuda.is_available())
print('  vllm', vllm.__version__)
print('  grpocredit OK')
from grpocredit.common.config import load_config
cfg = load_config('$ROOT/grpocredit/configs/oracle/rho1b_sft_gsm8k.yaml')
assert cfg.model.name_or_path == 'realtreetune/rho-1b-sft-GSM8K', cfg.model.name_or_path
print('  load_config(rho1b_sft_gsm8k.yaml) OK')
"
( cd "$ROOT/grpocredit" && pytest -q )
conda deactivate

# 4. HF cache: at least the rho-1b-sft-GSM8K checkpoint must be downloaded
#    (smallest one — fast iteration target). Other ckpts: warn if missing.
echo "[sanity] === HF cache ==="
HF_HOME_DEFAULT="${HF_HOME:-$HOME/.cache/huggingface}"
need_one() {
    local repo="$1" required="$2"
    local cache_dir="$HF_HOME_DEFAULT/hub/models--${repo//\//--}"
    if [ -d "$cache_dir" ]; then
        echo "  [ok]    $repo"
    else
        if [ "$required" = "required" ]; then
            echo "  [FAIL]  $repo  (run: huggingface-cli download $repo)" >&2
            return 1
        else
            echo "  [warn]  $repo  (run: huggingface-cli download $repo if you plan to use it)"
        fi
    fi
}
need_one "realtreetune/rho-1b-sft-GSM8K"          required
need_one "realtreetune/deepseekmath-7b-sft-GSM8K" optional
need_one "realtreetune/deepseekmath-7b-sft-MATH-v2" optional
need_one "Qwen/Qwen2.5-Math-7B-Instruct"          optional

echo
echo "[sanity] PASS — server2 looks ready."
echo "  Next:"
echo "    cd VinePPO && bash scripts/launch_server2_vineppo.sh rho1bSft2 GSM8K"
echo "    cd VinePPO && bash scripts/launch_server2_grpo.sh    rho1bSft2 GSM8K"
echo "    cd grpocredit && bash scripts/run_oracle_on_grpo_iter.sh rho1bSft2 GSM8K 0"
