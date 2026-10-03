"""Measure training speed and memory for a config on synthetic batches (no dataset needed).

    uv run aqyn bench --config configs/ljspeech_cfc.yaml
    uv run aqyn bench --config configs/ljspeech_cfc.yaml --frames 3000,6000,9000 --no-compile

Prints seconds per step, frames per second and peak GPU memory for each batch size,
so you can pick the largest ``data.max_frames_per_batch`` that fits.
"""

from __future__ import annotations

import argparse
import time
from contextlib import nullcontext

import torch

from .config import load_config
from .models import TTSModel, count_parameters
from .models.cfc import disable_compiled_step, enable_compiled_step
from .train import autocast_for

LJSPEECH_TRAIN_FRAMES = 1_030_000  # ~23 h at 12.5 frames/s


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--config", required=True)
    ap.add_argument("--frames", default="1500,3000,6000,9000,12000", help="batch sizes in frames")
    ap.add_argument("--utt-frames", type=int, default=100, help="frames per synthetic utterance")
    ap.add_argument("--steps", type=int, default=10, help="timed steps per batch size")
    ap.add_argument("--no-compile", action="store_true", help="disable the compiled CfC step")
    ap.add_argument("overrides", nargs="*")


def synthetic_batch(b: int, t: int, n: int, vocab: int, cfg, device) -> dict:
    n_words = n // 6
    word_frames = torch.arange(n_words, device=device) * (t // n_words)
    return {
        "word_starts": (torch.arange(n_words, device=device) * 6).expand(b, -1),
        "word_frames": word_frames.expand(b, -1),
        "num_words": torch.full((b,), n_words, device=device),
        "text": torch.randint(4, vocab, (b, n), device=device),
        "text_lens": torch.full((b,), n, device=device),
        "codes": torch.randint(0, cfg.codebook_size, (b, t, cfg.num_codebooks), device=device),
        "code_lens": torch.full((b,), t, device=device),
        "speaker": torch.zeros(b, dtype=torch.long, device=device),
    }


def run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config, args.overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        print(f"GPU: {torch.cuda.get_device_name()}")
    vocab = 40
    model = TTSModel(cfg.model, vocab, 0).to(device).train()
    print(f"model: {count_parameters(model)['total'] / 1e6:.1f}M params, layers {cfg.model.layers}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=device.type == "cuda")
    autocast, scaler = autocast_for(cfg.train.precision, device)

    disable_compiled_step()
    if not args.no_compile and device.type == "cuda" and "cfc" in cfg.model.layers:
        ok = enable_compiled_step(device, None if autocast is nullcontext else autocast)
        print(f"compiled CfC step: {'on' if ok else 'off'}")

    t, n = args.utt_frames, int(args.utt_frames * 1.2)
    print(
        f"\n{'frames':>7} {'batch':>5} {'s/step':>7} {'frames/s':>9} {'peak GB':>8} {'h / 100k steps':>15}"
    )
    for frames in (int(x) for x in args.frames.split(",")):
        b = max(1, frames // t)
        batch = synthetic_batch(b, t, n, vocab, cfg.model, device)
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            for i in range(args.steps + 3):  # 3 warm-up steps
                if i == 3:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                with autocast():
                    loss = model(batch, cfg.train)["loss"]
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / args.steps
        except torch.OutOfMemoryError:
            print(f"{b * t:>7} {b:>5}  out of memory")
            break
        peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else float("nan")
        print(
            f"{b * t:>7} {b:>5} {dt:>7.3f} {b * t / dt:>9.0f} {peak:>8.2f} {dt * 1e5 / 3600:>15.1f}"
        )
    print(
        f"\nOne LJSpeech epoch is ~{LJSPEECH_TRAIN_FRAMES:,} frames. Pick the largest batch that fits "
        "with ~1-2 GB to spare and set data.max_frames_per_batch."
    )
