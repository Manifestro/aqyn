"""Evaluate a checkpoint (or the Mimi ceiling) on a held-out split.

    # Ceiling: Mimi encode/decode of the real recordings, no model involved
    uv run aqyn eval --ceiling --data data/ljspeech_tokens --out results/mimi_ceiling

    # A trained model
    uv run aqyn eval --ckpt runs/ljspeech_cfc/best.pt --out results/ljspeech_cfc

Metrics: WER / CER of an ASR model on the audio, optional UTMOS, time to first frame
and generation real-time factor. Writes ``results.json`` and per-utterance ``utterances.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import soxr
import torch
from tqdm import tqdm

from .checkpoint import load_model
from .codec import FRAME_RATE, SAMPLE_RATE, Mimi, save_audio
from .data import TokenDataset


def norm_for_wer(text: str) -> str:
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^a-z' ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class ASR:
    def __init__(self, model_id: str, device: str):
        from transformers import pipeline

        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.pipe = pipeline(
            "automatic-speech-recognition", model=model_id, device=device, dtype=dtype
        )

    def __call__(self, wav: np.ndarray) -> str:
        wav16 = soxr.resample(wav, SAMPLE_RATE, 16_000)
        out = self.pipe(
            {"raw": wav16, "sampling_rate": 16_000},
            generate_kwargs={"language": "en", "task": "transcribe"},
        )
        return out["text"]


def load_utmos(device: str):
    try:
        model = (
            torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
            .to(device)
            .eval()
        )
    except Exception as e:
        print(f"UTMOS unavailable, skipping: {e}")
        return None

    @torch.no_grad()
    def score(wav: np.ndarray) -> float:
        x = torch.from_numpy(wav).float().unsqueeze(0).to(device)
        return float(model(x, SAMPLE_RATE))

    return score


def add_args(ap: argparse.ArgumentParser) -> None:
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt")
    src.add_argument("--ceiling", action="store_true")
    ap.add_argument(
        "--data", default=None, help="token dataset root (defaults to the checkpoint's)"
    )
    ap.add_argument("--split", default="test")
    ap.add_argument("--num", type=int, default=100, help="number of utterances")
    ap.add_argument("--out", required=True)
    ap.add_argument("--asr", default="openai/whisper-large-v3-turbo")
    ap.add_argument("--utmos", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


def run(args: argparse.Namespace) -> None:

    import jiwer

    torch.manual_seed(args.seed)
    model = vocab = None
    if args.ckpt:
        model, vocab, cfg, _ = load_model(args.ckpt, args.device)
        data_root = args.data or cfg.data.root
    else:
        if not args.data:
            raise SystemExit("--ceiling needs --data")
        data_root = args.data
    ds = TokenDataset(data_root, args.split)
    if vocab is not None and vocab.symbols != ds.vocab.symbols:
        print(
            "warning: checkpoint vocabulary differs from the dataset vocabulary; using the checkpoint's"
        )

    out = Path(args.out)
    (out / "wavs").mkdir(parents=True, exist_ok=True)
    mimi = Mimi(args.device)
    asr = ASR(args.asr, args.device)
    utmos = load_utmos(args.device) if args.utmos else None

    rows = []
    for i in tqdm(range(min(args.num, len(ds))), desc="eval"):
        item = ds[i]
        ref_text = ds.items[i]["text"]
        row = {"id": item["id"], "ref": ref_text}
        if model is None:
            codes = item["codes"]
        else:
            text = torch.tensor(vocab.encode(ref_text), device=args.device)
            first = {}
            t0 = time.perf_counter()

            def on_frame(t, _c, first=first, t0=t0):
                if t == 0:
                    if args.device.startswith("cuda"):
                        torch.cuda.synchronize()
                    first["t"] = time.perf_counter() - t0

            codes = model.generate(
                text, temperature=args.temperature, top_k=args.top_k, on_frame=on_frame
            )
            gen_s = time.perf_counter() - t0
            row["first_frame_ms"] = first["t"] * 1000
            row["rtf"] = gen_s / (len(codes) / FRAME_RATE)
        wav = mimi.decode(codes).numpy()
        save_audio(out / "wavs" / f"{item['id']}.wav", wav)
        row["seconds"] = len(codes) / FRAME_RATE
        row["ref_seconds"] = len(item["codes"]) / FRAME_RATE
        row["hyp"] = asr(wav)
        if utmos is not None:
            row["utmos"] = utmos(wav)
        rows.append(row)

    refs = [norm_for_wer(r["ref"]) for r in rows]
    hyps = [norm_for_wer(r["hyp"]) for r in rows]
    summary = {
        "source": "mimi_ceiling" if model is None else str(args.ckpt),
        "split": args.split,
        "n": len(rows),
        "wer": jiwer.wer(refs, hyps),
        "cer": jiwer.cer(refs, hyps),
        "duration_ratio": float(np.mean([r["seconds"] / r["ref_seconds"] for r in rows])),
    }
    for key in ("utmos", "first_frame_ms", "rtf"):
        vals = [r[key] for r in rows if key in r]
        if vals:
            summary[key] = float(np.mean(vals))

    with open(out / "utterances.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
