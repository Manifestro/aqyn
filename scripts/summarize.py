"""Collect everything under results/ and runs/ into one Markdown report.

    uv run python scripts/summarize.py > results/summary.md

Reads what ``scripts/pod/evaluate.sh`` writes: ``results/<backbone>_s<seed>/eval_seed*/``,
``.../longform/``, ``results/latency/`` and ``results/ceiling/``. WER intervals are 95%
bootstrap intervals over utterances; the comparison against the reference backbone is a
paired bootstrap over the utterances both models were scored on.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import jiwer
import numpy as np

from aqyn.evaluate import norm_for_wer

RNG = np.random.default_rng(0)
BOOT = 2000


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def errors(rows: list[dict]) -> dict[str, tuple[int, int]]:
    """Per utterance: (word errors, reference words)."""
    out = {}
    for r in rows:
        ref, hyp = norm_for_wer(r["ref"]), norm_for_wer(r["hyp"])
        if not ref:
            continue
        m = jiwer.process_words(ref, hyp)
        out[r["id"]] = (m.substitutions + m.deletions + m.insertions, len(ref.split()))
    return out


def wer_ci(err: np.ndarray, n: np.ndarray) -> tuple[float, float, float]:
    idx = RNG.integers(0, len(err), (BOOT, len(err)))
    boot = err[idx].sum(1) / n[idx].sum(1)
    return err.sum() / n.sum(), *np.percentile(boot, [2.5, 97.5])


def paired(a: dict, b: dict) -> tuple[float, float, float, float] | None:
    """WER(a) - WER(b) on shared utterances: difference, 95% interval, share of resamples > 0."""
    keys = sorted(set(a) & set(b))
    if len(keys) < 20:
        return None
    ea, eb = np.array([a[k][0] for k in keys]), np.array([b[k][0] for k in keys])
    n = np.array([a[k][1] for k in keys])
    idx = RNG.integers(0, len(keys), (BOOT, len(keys)))
    diff = (ea[idx].sum(1) - eb[idx].sum(1)) / n[idx].sum(1)
    return (
        (ea.sum() - eb.sum()) / n.sum(),
        *np.percentile(diff, [2.5, 97.5]),
        float((diff > 0).mean()),
    )


def mean(rows: list[dict], key: str) -> str:
    vals = [r[key] for r in rows if key in r]
    return f"{np.mean(vals):.3f}" if vals else "—"


def size(mb: float) -> str:
    return f"{mb:.1f} MB" if mb >= 1 else f"{mb * 1024:.0f} KB"


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--runs", default="runs")
    ap.add_argument(
        "--reference", default="transformer", help="backbone the others are compared to"
    )
    args = ap.parse_args()
    results, runs = Path(args.results), Path(args.runs)

    # --- test set -----------------------------------------------------------------
    per_run: dict[str, dict] = {}  # "cfc_s0" -> {"rows": [...], "err": {(id, sampling seed): ...}}
    for d in sorted(results.glob("*_s[0-9]*")):
        rows, err = [], {}
        for ev in sorted(d.glob("eval_seed*")):
            if not (ev / "utterances.jsonl").exists():
                continue
            part = read_jsonl(ev / "utterances.jsonl")
            rows += part
            err.update({(k, ev.name): v for k, v in errors(part).items()})
        if rows:
            per_run[d.name] = {"rows": rows, "err": err}

    print("# Backbone comparison\n")
    print("## Test set\n")
    print(
        "| Run | Utterances x seeds | WER (95% CI) | UTMOS | Spk-sim | Duration ratio | RTF (GPU) |"
    )
    print("|---|---|---|---|---|---|---|")
    ceiling = results / "ceiling" / "utterances.jsonl"
    if ceiling.exists():
        rows = read_jsonl(ceiling)
        e = errors(rows)
        w, lo, hi = wer_ci(*np.array(list(e.values())).T)
        print(
            f"| mimi ceiling | {len(rows)} | {pct(w)} ({pct(lo)}–{pct(hi)}) | {mean(rows, 'utmos')} | — | 1.000 | — |"
        )
    for name, r in per_run.items():
        w, lo, hi = wer_ci(*np.array(list(r["err"].values())).T)
        ratio = np.mean([x["seconds"] / x["ref_seconds"] for x in r["rows"]])
        print(
            f"| {name} | {len(r['rows'])} | {pct(w)} ({pct(lo)}–{pct(hi)}) | {mean(r['rows'], 'utmos')} "
            f"| {mean(r['rows'], 'spk_sim')} | {ratio:.3f} | {mean(r['rows'], 'rtf')} |"
        )

    by_backbone = defaultdict(dict)  # backbone -> pooled errors over training seeds
    for name, r in per_run.items():
        backbone, seed = re.match(r"(.+)_s(\d+)$", name).groups()
        by_backbone[backbone].update({(*k, seed): v for k, v in r["err"].items()})
    if args.reference in by_backbone and len(by_backbone) > 1:
        print(f"\n### WER difference to {args.reference} (paired bootstrap, all seeds pooled)\n")
        print(f"| Backbone | WER - WER({args.reference}) | 95% CI | P(worse) |")
        print("|---|---|---|---|")
        for backbone, err in by_backbone.items():
            if backbone == args.reference:
                continue
            p = paired(err, by_backbone[args.reference])
            if p:
                d, lo, hi, worse = p
                print(
                    f"| {backbone} | {100 * d:+.2f} pt | {100 * lo:+.2f} … {100 * hi:+.2f} pt | {worse:.2f} |"
                )
        print("\nAn interval that contains 0 means the test cannot tell the two backbones apart.")

    # --- long form ----------------------------------------------------------------
    long = {
        d.parent.parent.name: json.load(open(d))
        for d in sorted(results.glob("*_s[0-9]*/longform/results.json"))
    }
    if long:
        ks = sorted({s["k"] for r in long.values() for s in r["lengths"]})
        secs = {s["k"]: s["ref_seconds"] for r in long.values() for s in r["lengths"]}
        for title, key, fmt in [
            ("WER", "wer", pct),
            (
                "Voice drift (similarity of the first and last 4 s; higher is steadier)",
                "spk_drift",
                "{:.3f}".format,
            ),
            ("Share of generations that hit the frame limit", "truncated", "{:.2f}".format),
            ("Duration ratio", "duration_ratio", "{:.3f}".format),
        ]:
            print(f"\n## Long form: {title}\n")
            print("| Run | " + " | ".join(f"{k} sent. (~{secs[k]:.0f} s)" for k in ks) + " |")
            print("|---|" + "---|" * len(ks))
            for name, r in long.items():
                cells = {s["k"]: s for s in r["lengths"]}
                print(
                    f"| {name} | "
                    + " | ".join(fmt(cells[k][key]) if key in cells.get(k, {}) else "—" for k in ks)
                    + " |"
                )

    # --- streaming cost -----------------------------------------------------------
    lat = sorted((results / "latency").glob("*.json"))
    if lat:
        print("\n## Streaming cost\n")
        print(
            "| Backbone | Device | Stream | Backbone ms / frame | Last 50 frames | Depth ms | RTF | State | Peak memory |"
        )
        print("|---|---|---|---|---|---|---|---|---|")
        for path in lat:
            r = json.load(open(path))
            dev = r["device"] + (f", {r['threads']} thr" if r["threads"] else "")
            for s in r["streams"]:
                print(
                    f"| {path.stem.rsplit('_', 1)[0]} | {dev} | {s['seconds']:.0f} s | {s['backbone_ms']:.2f} "
                    f"| {s['backbone_ms_last50']:.2f} | {s['depth_ms']:.2f} | {s['rtf']:.3f} "
                    f"| {size(s['state_mb'])} | {s['peak_mb']:.0f} MB |"
                )

    # --- training -----------------------------------------------------------------
    logs = sorted(runs.glob("*/log.jsonl"))
    if logs:
        print("\n## Training\n")
        print(
            "| Run | Steps | s / step | Train codes | Val codes | Val advance acc. | Best val codes (step) |"
        )
        print("|---|---|---|---|---|---|---|")
        for path in logs:
            recs = read_jsonl(path)
            tr = [r for r in recs if r["split"] == "train"]
            va = [r for r in recs if r["split"] == "val"]
            if not tr or not va:
                continue
            best = min(va, key=lambda r: r["loss_codes"])
            print(
                f"| {path.parent.name} | {tr[-1]['step']} | {np.median([r['s_per_step'] for r in tr]):.2f} "
                f"| {tr[-1]['loss_codes']:.3f} | {va[-1]['loss_codes']:.3f} | {va[-1]['adv_acc']:.3f} "
                f"| {best['loss_codes']:.3f} ({best['step']}) |"
            )


if __name__ == "__main__":
    main()
