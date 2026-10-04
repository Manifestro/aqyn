"""Long-form test: intelligibility and voice stability as the text gets longer.

    uv run aqyn longform --ckpt runs/libritts_r_cfc/last.pt --out results/cfc_longform

Paragraphs are built by joining ``k`` consecutive test sentences (``--concat 1,2,4,...``) and
generated in one stream with the voice of the first sentence's speaker. Training clips are
at most 20 s, so anything beyond that is out of distribution: this is where a backbone
either holds on (the text comes through the window, the state stays bounded) or drifts.

Per length: WER / CER, duration relative to the summed references, share of generations
that ran into the frame limit, speaker similarity to the reference, and voice drift
(similarity between the first and the last ``--drift-seconds`` of the generated audio).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .checkpoint import load_model
from .codec import FRAME_RATE, SAMPLE_RATE, Mimi, save_audio
from .data import TokenDataset
from .evaluate import ASR, SpeakerSim, norm_for_wer, timed_generate


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--data", default=None, help="token dataset root (defaults to the checkpoint's)"
    )
    ap.add_argument("--split", default="test")
    ap.add_argument("--concat", default="1,2,4,8,16,48", help="sentences per paragraph")
    ap.add_argument("--num", type=int, default=30, help="paragraphs per length")
    ap.add_argument(
        "--max-seconds", type=float, default=1500.0, help="audio budget per length (reference)"
    )
    ap.add_argument("--asr", default="openai/whisper-large-v3-turbo")
    ap.add_argument("--language", default="en")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--max-ratio", type=float, default=2.0, help="frame limit / reference length")
    ap.add_argument("--drift-seconds", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


def run(args: argparse.Namespace) -> None:
    import jiwer

    torch.manual_seed(args.seed)
    model, vocab, cfg, _ = load_model(args.ckpt, args.device)
    ds = TokenDataset(args.data or cfg.data.root, args.split)
    out = Path(args.out)
    (out / "wavs").mkdir(parents=True, exist_ok=True)
    mimi = Mimi(args.device)
    asr = ASR(args.asr, args.device, args.language)
    spk_sim = SpeakerSim(args.device)
    edge = int(args.drift_seconds * SAMPLE_RATE)

    summaries, rows = [], []
    for k in (int(x) for x in args.concat.split(",")):
        groups = [ds.items[i : i + k] for i in range(0, len(ds.items) - k + 1, k)]
        mean_s = float(np.mean([sum(it["frames"] for it in g) for g in groups])) / FRAME_RATE
        groups = groups[: max(1, min(args.num, int(args.max_seconds / mean_s)))]
        k_rows = []
        for j, group in enumerate(tqdm(groups, desc=f"k={k}")):
            text = " ".join(it["text"] for it in group)
            ref_frames = sum(it["frames"] for it in group)
            limit = int(args.max_ratio * ref_frames) + 10
            codes, _, gen_s = timed_generate(
                model,
                vocab,
                text,
                group[0].get("speaker", 0),
                args.device,
                max_frames=limit,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            wav = mimi.decode(codes).numpy()
            save_audio(out / "wavs" / f"k{k:02d}_{j:03d}.wav", wav)
            first = group[0]
            ref_wav = mimi.decode(
                torch.from_numpy(
                    np.asarray(
                        ds.codes[first["offset"] : first["offset"] + first["frames"]],
                        dtype=np.int64,
                    )
                )
            ).numpy()
            row = {
                "k": k,
                "ref": text,
                "hyp": asr(wav),
                "seconds": len(codes) / FRAME_RATE,
                "ref_seconds": ref_frames / FRAME_RATE,
                "truncated": len(codes) >= limit,
                "rtf": gen_s / (len(codes) / FRAME_RATE),
                "spk_sim": spk_sim(wav, ref_wav),
            }
            if len(wav) >= 2 * edge:
                row["spk_drift"] = spk_sim(wav[:edge], wav[-edge:])
            k_rows.append(row)
        refs = [norm_for_wer(r["ref"]) for r in k_rows]
        hyps = [norm_for_wer(r["hyp"]) for r in k_rows]
        summary = {
            "k": k,
            "n": len(k_rows),
            "ref_seconds": float(np.mean([r["ref_seconds"] for r in k_rows])),
            "wer": jiwer.wer(refs, hyps),
            "cer": jiwer.cer(refs, hyps),
            "duration_ratio": float(np.mean([r["seconds"] / r["ref_seconds"] for r in k_rows])),
            "truncated": float(np.mean([r["truncated"] for r in k_rows])),
            "spk_sim": float(np.mean([r["spk_sim"] for r in k_rows])),
            "rtf": float(np.mean([r["rtf"] for r in k_rows])),
        }
        drift = [r["spk_drift"] for r in k_rows if "spk_drift" in r]
        if drift:
            summary["spk_drift"] = float(np.mean(drift))
        summaries.append(summary)
        rows += k_rows
        print(json.dumps(summary))

    with open(out / "utterances.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "results.json", "w", encoding="utf-8") as f:
        json.dump(
            {"source": str(args.ckpt), "split": args.split, "lengths": summaries}, f, indent=2
        )
