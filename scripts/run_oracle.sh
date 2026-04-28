#!/usr/bin/env bash
# Phase A oracle launcher for any one of the 3 policies.
#
# Usage:
#   bash scripts/run_oracle.sh configs/oracle/qwen_math_instruct.yaml
#   bash scripts/run_oracle.sh configs/oracle/deepseek_math_sft.yaml
#   bash scripts/run_oracle.sh configs/oracle/rho1b_sft_gsm8k.yaml
#
# Or run all three sequentially (server2 single-GPU; total wall-clock dominated
# by the two 7B configs):
#   for c in configs/oracle/*.yaml; do bash scripts/run_oracle.sh "$c"; done
#
# Output: each config writes to its own `output_dir` (set in the YAML), so the
# three runs do not clobber each other. Exit codes from sprint_d3:
#   0 proceed · 2 proceed-w-caveats · 3 pilot-required · 4 pivot-to-Plan-B · 5 missing data
#
# Env-vars (optional):
#   PROCEED_ON_POLICY_GATE_FAIL=1  — waive Day-1 policy-class failures
#       (short CoTs, §5 gate) but NOT infra failures. For intentionally-
#       weak debug policies, e.g. rho-1b. Failure is still recorded in
#       day1_summary.json / wandb / GATE_REPORT.md. See runbook §2.5.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <config-yaml> [n_trajectories=100] [verifier_probe_size=200]" >&2
    exit 64
fi

CONFIG="$1"
N_TRAJ="${2:-100}"
PROBE="${3:-200}"
PYTHON="${PYTHON:-python}"

DAY1_EXTRA_ARGS=()
if [ "${PROCEED_ON_POLICY_GATE_FAIL:-0}" = "1" ]; then
    DAY1_EXTRA_ARGS+=(--proceed-on-policy-gate-fail)
    echo "[oracle] PROCEED_ON_POLICY_GATE_FAIL=1 — Day 1 policy gate is advisory."
fi

# Read output_dir from the YAML so the four sprint scripts agree.
# Falls back to experiments/oracle/<config-stem> if not set in the YAML.
OUT="$($PYTHON - <<EOF
import sys
from grpocredit.common.config import load_config
cfg = load_config("$CONFIG")
print(cfg.output_dir)
EOF
)"

echo "[oracle] config=$CONFIG  output_dir=$OUT  n_traj=$N_TRAJ"
mkdir -p "$OUT"

# Tee a one-line provenance record so the runbook can grep for it later.
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) | $CONFIG | n_traj=$N_TRAJ probe=$PROBE" >> "$OUT/launch.log"

echo "[oracle] Day 1 — infrastructure smoke test"
# The `+` expansion below guards against `set -u` tripping on an empty array
# in older bashes.
"$PYTHON" scripts/sprint_d1_infra_smoke.py \
    --config "$CONFIG" \
    --n-trajectories "$N_TRAJ" \
    --verifier-probe-size "$PROBE" \
    --output-dir "$OUT" \
    ${DAY1_EXTRA_ARGS[@]+"${DAY1_EXTRA_ARGS[@]}"}

echo "[oracle] Day 2A — concordance check"
"$PYTHON" scripts/sprint_d2_concordance.py \
    --config "$CONFIG" \
    --n-trajectories "$N_TRAJ" \
    --output-dir "$OUT"

echo "[oracle] Day 2B — Q-variance oracle"
"$PYTHON" scripts/sprint_d2_oracle.py \
    --config "$CONFIG" \
    --output-dir "$OUT"

echo "[oracle] Day 3 — gate report"
"$PYTHON" scripts/sprint_d3_gate_report.py \
    --sprint-dir "$OUT" \
    --config "$CONFIG"

echo "[oracle] Done — see $OUT/GATE_REPORT.md"
