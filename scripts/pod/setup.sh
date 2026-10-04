#!/bin/bash
# One-time pod setup: code, environment, LibriTTS-R download and tokenization. Safe to re-run.
#   git clone https://github.com/Manifestro/aqyn.git /workspace/aqyn
#   bash /workspace/aqyn/scripts/pod/setup.sh
set -euo pipefail
source "$(dirname "$0")/common.sh"

if [ ! -d "$AQYN_ROOT/.git" ]; then
  git clone https://github.com/Manifestro/aqyn.git "$AQYN_ROOT"
fi
cd "$AQYN_ROOT"
git pull --ff-only
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --extra eval

if [ ! -f "$DATA/train_codes.npy" ]; then
  mkdir -p data/raw
  for s in train_clean_100 train_clean_360; do
    d="data/raw/LibriTTS_R/${s//_/-}"
    if [ ! -f "$d/.extracted" ]; then
      wget -c -q --show-progress -O "data/raw/$s.tar.gz" "$MIRROR/$s.tar.gz"
      tar xzf "data/raw/$s.tar.gz" -C data/raw --no-same-owner
      touch "$d/.extracted"
      rm "data/raw/$s.tar.gz"
    fi
  done
  "$AQYN" prepare --dataset libritts_r --subsets train-clean-100,train-clean-360 \
    --out "$DATA" --num-val 300 --num-test 500 --workers 24
fi
echo "setup done: $(wc -l < "$DATA/train.jsonl") training utterances in $DATA"
