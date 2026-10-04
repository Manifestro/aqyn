#!/bin/bash
# Train one backbone with the fixed comparison recipe; resumes if the run already has last.pt.
#   scripts/pod/train.sh <cfc|transformer|lstm|hybrid> [seed]
set -uo pipefail
source "$(dirname "$0")/common.sh"
cd "$AQYN_ROOT"
B="$1"; SEED="${2:-0}"; R="$(run_dir "$B" "$SEED")"
mkdir -p "$R"
if grep -q "^done: $STEPS steps" "$R/train.log" 2>/dev/null; then
  echo "$R is already trained"; exit 0
fi
RESUME=(); [ -f "$R/last.pt" ] && RESUME=(--resume "$R/last.pt")
echo "[launch $(stamp)] ${RESUME[*]:-}" >> "$R/train.log"
PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 "$AQYN" train --config "configs/libritts_r_$B.yaml" \
  data.root="$DATA" data.max_frames_per_batch="$FRAMES" train.out_dir="$R" train.seed="$SEED" \
  train.max_steps="$STEPS" train.warmup_steps=1000 train.eval_every=1000 train.save_every=500 \
  train.sample_every=2000 ${RESUME[@]+"${RESUME[@]}"} >> "$R/train.log" 2>&1
echo "[exit code $? at $(stamp)]" >> "$R/train.log"
# A run stopped with Ctrl-C exits cleanly but is not finished.
grep -q "^done: $STEPS steps" "$R/train.log"
