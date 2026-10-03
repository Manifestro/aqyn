"""Download LJSpeech, resample to 24 kHz, and encode it with frozen Mimi.

    uv run aqyn prepare --out data/ljspeech_tokens

Writes ``vocab.json``, ``{train,val,test}.jsonl`` and ``{split}_codes.npy`` (see aqyn/data.py).
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

from .align import CTC_SAMPLE_RATE, CTCAligner
from .codec import FRAME_RATE, Mimi, load_audio
from .text import CharVocab, normalize_text, words_of

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


def word_start_frames(spans: list[tuple[float, float]], frames: int) -> list[int]:
    """Mimi frame in which each word starts (non-decreasing, inside the utterance)."""
    out, last = [], 0
    for start, _ in spans:
        last = min(max(last, int(start * FRAME_RATE)), frames - 1)
        out.append(last)
    return out


def add_args(ap: argparse.ArgumentParser) -> None:
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
        "--align-model",
        default="facebook/wav2vec2-base-960h",
        help="character CTC model used for word alignment",
    )
    ap.add_argument(
        "--limit", type=int, default=None, help="only use the first N utterances (debugging)"
    )


def run(args: argparse.Namespace) -> None:

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
    aligner = CTCAligner(args.align_model, args.device)
    for split, split_items in splits.items():
        chunks, offset, failed = [], 0, 0
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for it in tqdm(split_items, desc=split):
                path = str(root / "wavs" / f"{it['id']}.wav")
                words = words_of(it["text"])
                spans = aligner.align(load_audio(path, CTC_SAMPLE_RATE), CTC_SAMPLE_RATE, words)
                if spans is None:
                    failed += 1
                    continue
                codes = mimi.encode(load_audio(path)).numpy().astype(np.uint16)
                rec = {
                    "id": it["id"],
                    "text": it["text"],
                    "offset": offset,
                    "frames": len(codes),
                    "speaker": 0,
                    "word_frames": word_start_frames(spans, len(codes)),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                chunks.append(codes)
                offset += len(codes)
        empty = np.zeros((0, mimi.num_codebooks), dtype=np.uint16)
        np.save(out / f"{split}_codes.npy", np.concatenate(chunks, axis=0) if chunks else empty)
        hours = offset / 12.5 / 3600
        print(
            f"{split}: {len(chunks)} utterances, {offset} frames ({hours:.2f} h), "
            f"{failed} dropped (alignment failed)"
        )
