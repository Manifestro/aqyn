"""Synthesize speech from text with a trained checkpoint.

uv run aqyn synth --ckpt runs/ljspeech_cfc/best.pt --text "Hello world." --out hello.wav
"""

from __future__ import annotations

import argparse
import time

import torch

from .checkpoint import load_model
from .codec import FRAME_RATE, Mimi, save_audio


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--max-frames", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


def run(args: argparse.Namespace) -> None:

    torch.manual_seed(args.seed)
    model, vocab, _, _ = load_model(args.ckpt, args.device)
    mimi = Mimi(args.device)
    ids, word_starts = vocab.encode(args.text)
    text = torch.tensor(ids, device=args.device)
    word_starts = torch.tensor(word_starts, device=args.device)

    first = {}
    t0 = time.perf_counter()

    def on_row(t, _row):
        if t == 0:
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            first["t"] = time.perf_counter() - t0

    codes = model.generate(
        text,
        word_starts,
        max_frames=args.max_frames,
        temperature=args.temperature,
        top_k=args.top_k,
        on_row=on_row,
    )
    gen_s = time.perf_counter() - t0
    wav = mimi.decode(codes)
    save_audio(args.out, wav)
    dur = len(codes) / FRAME_RATE
    print(
        f"{args.out}: {dur:.2f} s audio, {len(codes)} frames | first frame {first['t'] * 1000:.0f} ms | "
        f"generation RTF {gen_s / dur:.3f} (Mimi decoding not included)"
    )
