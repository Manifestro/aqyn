# Shared settings for the pod scripts. Source it; do not run it.
# Everything lives under AQYN_ROOT; put that on a persistent volume (e.g. /workspace on RunPod),
# otherwise checkpoints are lost when the pod stops.
AQYN_ROOT="${AQYN_ROOT:-/workspace/aqyn}"
DATA="${DATA:-data/libritts_r_tokens}"
FRAMES="${FRAMES:-26000}"      # Mimi frames per batch (26k needs a 48 GB GPU)
STEPS="${STEPS:-20000}"
MIRROR="${MIRROR:-https://openslr.elda.org/resources/141}"   # or https://www.openslr.org/resources/141
AQYN="$AQYN_ROOT/.venv/bin/aqyn"

run_dir() { echo "runs/libritts_r_${1}_s${2:-0}"; }       # run_dir <backbone> [seed]
stamp() { date -u +%FT%TZ; }
