#!/bin/bash
# Run on your own machine: copy results, logs and samples from the pod into this checkout.
#   scripts/pod/fetch.sh <user@ssh.runpod.io> [ssh key]        (WEIGHTS=1 also copies model-only last.pt)
# RunPod's proxy SSH has no scp/sftp, so the files travel as base64 through the terminal.
# With a direct TCP SSH port, plain rsync is faster.
set -euo pipefail
TARGET="$1"; KEY="${2:-$HOME/.ssh/id_ed25519}"
REMOTE_ROOT="${AQYN_ROOT:-/workspace/aqyn}"
cd "$(dirname "$0")/../.."
WHAT='--exclude=wavs results runs/*/log.jsonl runs/*/train.log runs/*/config.yaml runs/*/samples runs/queue.log'
PRE=':'
if [ "${WEIGHTS:-0}" = 1 ]; then
  PRE='for r in runs/libritts_r_*; do [ -f $r/last.pt ] && [ ! -f $r/model.pt ] && .venv/bin/python -c "import torch,sys; c=torch.load(sys.argv[1]+\"/last.pt\",map_location=\"cpu\",weights_only=False); c[\"optimizer\"]=None; torch.save(c,sys.argv[1]+\"/model.pt\")" $r; done'
  WHAT="$WHAT runs/*/model.pt"
fi
tmp="$(mktemp)"
printf 'stty -echo; cd %s; %s; echo BEGIN_B64; tar czf - %s 2>/dev/null | base64 -w 100; echo END_B64\nexit\n' \
  "$REMOTE_ROOT" "$PRE" "$WHAT" \
  | ssh -tt -o ServerAliveInterval=15 -i "$KEY" "$TARGET" 2>&1 | tr -d '\r' \
  | sed -n '/BEGIN_B64$/,/^END_B64$/p' | grep -E '^[A-Za-z0-9+/=]+$' | base64 -d > "$tmp"
tar xzf "$tmp" && rm "$tmp"
echo "fetched into $(pwd): results/, runs/"
