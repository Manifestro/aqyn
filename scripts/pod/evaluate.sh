#!/bin/bash
# Full evaluation of one trained run: test-set metrics with two sampling seeds, long-form test,
# streaming cost on CPU (1 thread) and GPU. Finished parts are skipped.
#   scripts/pod/evaluate.sh <cfc|transformer|lstm|hybrid> [seed]
set -uo pipefail
source "$(dirname "$0")/common.sh"
cd "$AQYN_ROOT"
B="$1"; SEED="${2:-0}"; R="$(run_dir "$B" "$SEED")"; OUT="results/${B}_s${SEED}"
CKPT="$R/last.pt"
[ -f "$CKPT" ] || { echo "no $CKPT"; exit 1; }
mkdir -p "$OUT" results/latency
L="$OUT/eval.log"

if [ ! -f results/ceiling/results.json ]; then
  "$AQYN" eval --ceiling --data "$DATA" --out results/ceiling --num 100000 --utmos >> "$L" 2>&1
fi
for s in 0 1; do
  [ -f "$OUT/eval_seed$s/results.json" ] || "$AQYN" eval --ckpt "$CKPT" --out "$OUT/eval_seed$s" \
    --num 100000 --seed "$s" --utmos --spk-sim >> "$L" 2>&1
done
[ -f "$OUT/longform/results.json" ] || "$AQYN" longform --ckpt "$CKPT" --out "$OUT/longform" >> "$L" 2>&1
for dev in cpu cuda; do
  [ -f "results/latency/${B}_$dev.json" ] || "$AQYN" latency --config "configs/libritts_r_$B.yaml" \
    --device "$dev" --threads 1 --out "results/latency/${B}_$dev.json" >> "$L" 2>&1
done
rm -rf "$OUT"/eval_seed*/wavs   # keep the long-form audio, drop ~1000 short clips
echo "[evaluated $B seed $SEED at $(stamp)]" >> "$L"
