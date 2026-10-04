#!/bin/bash
# Train and evaluate a list of runs one after another, detached from the terminal.
#   scripts/pod/queue.sh transformer:0 cfc:0 lstm:0          (backbone:seed ...)
# Progress: tail -f $AQYN_ROOT/runs/queue.log    Stop the current training: pkill -INT -f "aqyn train"
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
source "$(dirname "$0")/common.sh"
cd "$AQYN_ROOT"
mkdir -p runs results
if [ "${1:-}" != "--foreground" ]; then
  setsid nohup "$SELF" --foreground "$@" >> runs/queue.log 2>&1 < /dev/null &
  echo "queue started in the background; log: $AQYN_ROOT/runs/queue.log"
  exit 0
fi
shift
for job in "$@"; do
  B="${job%%:*}"; SEED="${job##*:}"
  echo "[$(stamp)] train $B seed $SEED"
  if scripts/pod/train.sh "$B" "$SEED"; then
    echo "[$(stamp)] evaluate $B seed $SEED"
    scripts/pod/evaluate.sh "$B" "$SEED" || echo "[$(stamp)] evaluation of $job failed"
  else
    echo "[$(stamp)] training of $job failed or was stopped; see $(run_dir "$B" "$SEED")/train.log"
  fi
  .venv/bin/python scripts/summarize.py > results/summary.md 2>> runs/queue.log || true
done
echo "[$(stamp)] QUEUE_DONE"
