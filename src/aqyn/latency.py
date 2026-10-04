"""Streaming cost of a model: time per frame, real-time factor, memory, per-stream state.

    uv run aqyn latency --config configs/libritts_r_cfc.yaml --device cpu --threads 1
    uv run aqyn latency --config configs/libritts_r_transformer.yaml --device cuda

Cost does not depend on the weights, so a config is enough (``--ckpt`` also works). Each
stream length is generated frame by frame, the way ``aqyn synth`` does, with the backbone
step and the depth module (the 8 codebooks of a frame) timed separately. Recurrent backbones
keep a fixed-size state; attention keeps a key/value cache that grows with every frame, which
shows up as a larger state and a slower last frame on long streams.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_model
from .codec import FRAME_RATE
from .config import load_config
from .models import TTSModel, count_parameters

VOCAB, PAD = 40, 0
FRAMES_PER_WORD, CHARS_PER_WORD = 4, 6


def state_bytes(state) -> int:
    """Bytes held by a (nested) mixer state."""
    if torch.is_tensor(state):
        return state.numel() * state.element_size()
    if isinstance(state, (list, tuple)):
        return sum(state_bytes(s) for s in state)
    return 0


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def stream(model: TTSModel, frames: int, device: torch.device) -> dict:
    """Generate ``frames`` rows for a synthetic text; per-frame times in ms and final state size."""
    words = frames // FRAMES_PER_WORD + 1
    text = torch.randint(4, VOCAB, (1, words * CHARS_PER_WORD), device=device)
    word_starts = (torch.arange(words, device=device) * CHARS_PER_WORD)[None]
    speaker = torch.zeros(1, dtype=torch.long, device=device)
    state = model.start(
        text,
        torch.tensor([text.shape[1]], device=device),
        word_starts,
        torch.tensor([words], device=device),
        speaker,
    )
    backbone, depth, prev = [], [], None
    for s in range(frames):
        _sync(device)
        t0 = time.perf_counter()
        h, _, _ = model.step(prev, state)
        _sync(device)
        t1 = time.perf_counter()
        prev = model.depth.sample(h)
        _sync(device)
        t2 = time.perf_counter()
        backbone.append((t1 - t0) * 1000)
        depth.append((t2 - t1) * 1000)
        if s % FRAMES_PER_WORD == FRAMES_PER_WORD - 1:
            state.pointer = (state.pointer + 1).clamp(max=words - 1)
    return {
        "backbone": np.array(backbone),
        "depth": np.array(depth),
        "state_bytes": state_bytes(state.mixer_states),
    }


def peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 2**10  # bytes on macOS, KiB on Linux


def add_args(ap: argparse.ArgumentParser) -> None:
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", help="YAML config (random weights)")
    src.add_argument("--ckpt", help="trained checkpoint")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1, help="CPU threads")
    ap.add_argument(
        "--frames", default="125,750,3750", help="stream lengths in frames (12.5 per second)"
    )
    ap.add_argument("--out", default=None, help="write the results as JSON here")


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.threads)
    if args.ckpt:
        model, _, cfg, _ = load_model(args.ckpt, device)
        mcfg = cfg.model
    else:
        mcfg = load_config(args.config).model
        model = TTSModel(mcfg, VOCAB, PAD).to(device).eval()
    params = count_parameters(model)["total"]
    print(
        f"{args.config or args.ckpt}: {params / 1e6:.1f}M params, layers {mcfg.layers}, "
        f"{device.type}" + (f", {args.threads} thread(s)" if device.type == "cpu" else "")
    )
    stream(model, 20, device)  # warm-up

    frame_ms = 1000 / FRAME_RATE
    rows = []
    print(
        f"\n{'frames':>7} {'audio s':>8} {'backbone ms':>12} {'first 50':>9} {'last 50':>8} "
        f"{'depth ms':>9} {'RTF':>6} {'state MB':>9} {'peak MB':>8}"
    )
    for frames in (int(x) for x in args.frames.split(",")):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        r = stream(model, frames, device)
        bb, dp = r["backbone"], r["depth"]
        row = {
            "frames": frames,
            "seconds": frames / FRAME_RATE,
            "backbone_ms": float(bb.mean()),
            "backbone_ms_first50": float(bb[:50].mean()),
            "backbone_ms_last50": float(bb[-50:].mean()),
            "backbone_ms_p95": float(np.percentile(bb, 95)),
            "depth_ms": float(dp.mean()),
            "rtf": float((bb.mean() + dp.mean()) / frame_ms),
            "rtf_backbone": float(bb.mean() / frame_ms),
            "state_mb": r["state_bytes"] / 2**20,
            "peak_mb": peak_memory_mb(device),
        }
        rows.append(row)
        print(
            f"{frames:>7} {row['seconds']:>8.0f} {row['backbone_ms']:>12.2f} "
            f"{row['backbone_ms_first50']:>9.2f} {row['backbone_ms_last50']:>8.2f} "
            f"{row['depth_ms']:>9.2f} {row['rtf']:>6.3f} {row['state_mb']:>9.3f} {row['peak_mb']:>8.0f}"
        )
    print(
        "\nbackbone ms: mean time of one backbone step; depth ms: the 8 codebooks of a frame "
        "(same module for every backbone).\nRTF = (backbone + depth) / 80 ms, Mimi decoding not "
        "included. On CPU, peak MB is the peak resident memory of the whole process."
    )
    if args.out:
        out = {
            "source": args.config or args.ckpt,
            "layers": mcfg.layers,
            "params": params,
            "device": device.type,
            "threads": args.threads if device.type == "cpu" else None,
            "streams": rows,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
