"""Download LJSpeech, resample to 24 kHz, and encode it with frozen Mimi.

    python scripts/prepare_ljspeech.py --out data/ljspeech_tokens

Writes ``vocab.json``, ``{train,val,test}.jsonl`` and ``{split}_codes.npy`` (see mimicfc/data.py).
The split is a fixed random split (seed 1234): 500 test, 100 val, the rest train.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from mimicfc.codec import Mimi, load_audio
from mimicfc.text import CharVocab, normalize_text

URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        with tqdm(total=total, unit="B", unit_scale=True, desc="download") as bar:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.replace(dest)


def ensure_ljspeech(raw_dir: Path) -> Path:
    root = raw_dir / "LJSpeech-1.1"
    if (root / "metadata.csv").exists():
        return root
    archive = raw_dir / "LJSpeech-1.1.tar.bz2"
    if not archive.exists():
        print(f"Downloading LJSpeech (~2.6 GB) to {archive}")
        download(URL, archive)
    print("Extracting...")
    with tarfile.open(archive) as tar:
        tar.extractall(raw_dir)
    return root


def read_metadata(root: Path) -> list[dict]:
    items = []
    with open(root / "metadata.csv", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE):
            text = row[2] if len(row) > 2 and row[2].strip() else row[1]
            items.append({"id": row[0], "text": text})
    return items


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default="data/ljspeech_tokens")
    ap.add_argument(
        "--raw-dir", default="data/raw", help="where LJSpeech is (or will be) downloaded"
    )
    ap.add_argument(
        "--ljspeech-dir", default=None, help="existing extracted LJSpeech-1.1 directory"
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num-test", type=int, default=500)
    ap.add_argument("--num-val", type=int, default=100)
    ap.add_argument(
        "--limit", type=int, default=None, help="only use the first N utterances (debugging)"
    )
    args = ap.parse_args()

    root = Path(args.ljspeech_dir) if args.ljspeech_dir else ensure_ljspeech(Path(args.raw_dir))
    items = read_metadata(root)
    items = [it for it in items if normalize_text(it["text"])]
    random.Random(1234).shuffle(items)
    if args.limit:
        items = items[: args.limit]
    n_test = min(args.num_test, len(items) // 5)
    n_val = min(args.num_val, len(items) // 5)
    splits = {
        "test": items[:n_test],
        "val": items[n_test : n_test + n_val],
        "train": items[n_test + n_val :],
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vocab = CharVocab.build([it["text"] for it in splits["train"]])
    vocab.save(out / "vocab.json")
    print(f"vocab: {len(vocab)} symbols")

    mimi = Mimi(args.device)
    for split, split_items in splits.items():
        chunks, offset = [], 0
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for it in tqdm(split_items, desc=split):
                wav = load_audio(str(root / "wavs" / f"{it['id']}.wav"))
                codes = mimi.encode(wav).numpy().astype(np.uint16)
                rec = {
                    "id": it["id"],
                    "text": it["text"],
                    "offset": offset,
                    "frames": len(codes),
                    "speaker": 0,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                chunks.append(codes)
                offset += len(codes)
        np.save(out / f"{split}_codes.npy", np.concatenate(chunks, axis=0))
        hours = offset / 12.5 / 3600
        print(f"{split}: {len(split_items)} utterances, {offset} frames ({hours:.2f} h)")


if __name__ == "__main__":
    main()
