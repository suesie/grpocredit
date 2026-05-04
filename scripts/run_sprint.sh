#!/usr/bin/env bash
# Full sprint — run Days 1-3 in order with sensible defaults. Exits non-zero if
# any gate fails; use the exit code to branch into Plan B in automation.

set -euo pipefail

CONFIG="${CONFIG:-configs/base_qwen_math.yaml}"
OUT="${OUT:-experiments/sprint}"
PYTHON="${PYTHON:-python}"

echo "[sprint] Day 1 — infrastructure smoke test"
"$PYTHON" scripts/sprint_d1_infra_smoke.py \
    --config "$CONFIG" \
    --n-trajectories 100 \
    --verifier-probe-size 200 \
    --output-dir "$OUT"

echo "[sprint] Day 2A — embedding-variance diagnostic"
"$PYTHON" scripts/sprint_d2_concordance.py \
    --config "$CONFIG" \
    --output-dir "$OUT"

echo "[sprint] Day 2B — Q-variance oracle"
"$PYTHON" scripts/sprint_d2_oracle.py \
    --config "$CONFIG" \
    --output-dir "$OUT"

echo "[sprint] Day 3 — gate report"
"$PYTHON" scripts/sprint_d3_gate_report.py \
    --sprint-dir "$OUT" \
    --config "$CONFIG"
