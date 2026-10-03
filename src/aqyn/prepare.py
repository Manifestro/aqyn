"""Download a corpus, resample to 24 kHz, and encode it with frozen Mimi.

    uv run aqyn prepare --out data/ljspeech_tokens
    uv run aqyn prepare --dataset libritts_r --out data/libritts_r_tokens

Writes ``vocab.json``, ``{train,val,test}.jsonl`` and ``{split}_codes.npy`` (see aqyn/data.py).
The split is a fixed random split over utterances (seed 1234): 500 test, 100 val, the rest
train. For LibriTTS-R every speaker is therefore seen in training; ``speakers.json`` lists
the corpus speaker ids in the order of the ``speaker`` index stored in the manifests.
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
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .align import CTC_SAMPLE_RATE, CTCAligner
from .codec import FRAME_RATE, Mimi, load_audio
from .text import CharVocab, normalize_text, words_of

URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
LIBRITTS_R_URL = "https://www.openslr.org/resources/141/{name}.tar.gz"


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
            path = str(root / "wavs" / f"{row[0]}.wav")
            items.append({"id": row[0], "text": text, "path": path, "speaker": 0})
    return items


def ensure_libritts_r(raw_dir: Path, subsets: list[str]) -> Path:
    root = raw_dir / "LibriTTS_R"
    for subset in subsets:
        if (root / subset).is_dir():
            continue
        name = subset.replace("-", "_")
        archive = raw_dir / f"{name}.tar.gz"
        if not archive.exists():
            print(f"Downloading LibriTTS-R {subset} to {archive}")
            download(LIBRITTS_R_URL.format(name=name), archive)
        print(f"Extracting {subset}...")
        with tarfile.open(archive) as tar:
            tar.extractall(raw_dir)
    return root


def read_libritts_r(root: Path, subsets: list[str]) -> tuple[list[dict], list[str]]:
    """Utterances of ``root/<subset>/<speaker>/<chapter>/*.wav`` and the sorted speaker ids."""
    found = []
    for subset in subsets:
        for wav in sorted((root / subset).glob("*/*/*.wav")):
            txt = wav.with_suffix(".normalized.txt")
            if txt.exists():
                found.append((wav, txt.read_text(encoding="utf-8").strip()))
    speakers = sorted({wav.parts[-3] for wav, _ in found}, key=int)
    index = {s: i for i, s in enumerate(speakers)}
    items = [
        {"id": wav.stem, "text": text, "path": str(wav), "speaker": index[wav.parts[-3]]}
        for wav, text in found
    ]
    return items, speakers


class _Audio(Dataset):
    """Loads each utterance at the Mimi and CTC sample rates (in DataLoader workers)."""

    def __init__(self, items: list[dict], min_seconds: float, max_seconds: float):
        self.items, self.min_seconds, self.max_seconds = items, min_seconds, max_seconds

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path = self.items[i]["path"]
        info = sf.info(path)
        if not self.min_seconds <= info.frames / info.samplerate <= self.max_seconds:
            return None
        return load_audio(path), load_audio(path, CTC_SAMPLE_RATE)


def _identity(x):
    return x


def word_start_frames(spans: list[tuple[float, float]], frames: int) -> list[int]:
    """Mimi frame in which each word starts (non-decreasing, inside the utterance)."""
    out, last = [], 0
    for start, _ in spans:
        last = min(max(last, int(start * FRAME_RATE)), frames - 1)
        out.append(last)
    return out


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--dataset", default="ljspeech", choices=["ljspeech", "libritts_r"])
    ap.add_argument("--out", default="data/ljspeech_tokens")
    ap.add_argument(
        "--raw-dir", default="data/raw", help="where the corpus is (or will be) downloaded"
    )
    ap.add_argument(
        "--subsets",
        default="train-clean-100,train-clean-360",
        help="LibriTTS-R subsets to pool (comma separated)",
    )
    ap.add_argument("--min-seconds", type=float, default=0.5, help="skip shorter utterances")
    ap.add_argument("--max-seconds", type=float, default=20.0, help="skip longer utterances")
    ap.add_argument("--workers", type=int, default=8, help="audio loading processes")
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

    speakers = None
    if args.dataset == "libritts_r":
        subsets = [s for s in args.subsets.split(",") if s]
        items, speakers = read_libritts_r(ensure_libritts_r(Path(args.raw_dir), subsets), subsets)
        print(f"LibriTTS-R: {len(items)} utterances, {len(speakers)} speakers")
    else:
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
    if speakers is not None:
        with open(out / "speakers.json", "w", encoding="utf-8") as f:
            json.dump(speakers, f)

    mimi = Mimi(args.device)
    aligner = CTCAligner(args.align_model, args.device)
    for split, split_items in splits.items():
        chunks, offset, failed, skipped = [], 0, 0, 0
        # LJSpeech has no length limits: its clips are all 1-10 s.
        limits = (args.min_seconds, args.max_seconds) if speakers is not None else (0.0, 1e9)
        loader = DataLoader(
            _Audio(split_items, *limits),
            batch_size=None,
            num_workers=args.workers,
            collate_fn=_identity,
        )
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for it, audio in zip(
                split_items, tqdm(loader, desc=split, mininterval=30), strict=True
            ):
                if audio is None:
                    skipped += 1
                    continue
                wav, wav_ctc = audio
                words = words_of(it["text"])
                spans = aligner.align(wav_ctc, CTC_SAMPLE_RATE, words)
                if spans is None:
                    failed += 1
                    continue
                codes = mimi.encode(wav).numpy().astype(np.uint16)
                rec = {
                    "id": it["id"],
                    "text": it["text"],
                    "offset": offset,
                    "frames": len(codes),
                    "speaker": it["speaker"],
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
            f"{failed} dropped (alignment failed), {skipped} skipped (length)"
        )
