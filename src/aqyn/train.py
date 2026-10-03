"""Train a model from a YAML config.

uv run aqyn train --config configs/ljspeech_cfc.yaml
uv run aqyn train --config configs/ljspeech_cfc.yaml data.max_frames_per_batch=6000
uv run aqyn train --config configs/ljspeech_cfc.yaml --resume runs/ljspeech_cfc/last.pt
"""

from __future__ import annotations

import argparse
import json
import math
import random
import signal
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint
from .config import Config, load_config
from .data import FrameBudgetSampler, TokenDataset, make_collate
from .models import TTSModel, count_parameters
from .models.cfc import enable_compiled_step


def lr_at(step: int, cfg) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps))
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def autocast_for(precision: str, device: torch.device):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext, None
    if precision == "bf16" and torch.cuda.is_bf16_supported():
        return lambda: torch.autocast("cuda", dtype=torch.bfloat16), None
    return lambda: torch.autocast("cuda", dtype=torch.float16), torch.amp.GradScaler("cuda")


def to_device(batch: dict, device: torch.device) -> dict:
    return {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()
    }


@torch.no_grad()
def evaluate(model, loader, tcfg, device, autocast) -> dict[str, float]:
    model.eval()
    totals, n = {}, 0
    for batch in loader:
        with autocast():
            out = model(to_device(batch, device), tcfg)
        for k, v in out.items():
            totals[k] = totals.get(k, 0.0) + float(v)
        n += 1
    model.train()
    return {k: v / max(1, n) for k, v in totals.items()}


def write_samples(model, dataset, out_dir: Path, step: int, device, mimi_holder: dict, n: int = 2):
    from .codec import Mimi, save_audio

    if "mimi" not in mimi_holder:
        try:
            mimi_holder["mimi"] = Mimi(device)
        except Exception as e:  # sampling is optional; never kill a training run over it
            print(f"[samples] could not load Mimi, disabling samples: {e}")
            mimi_holder["mimi"] = None
    mimi = mimi_holder["mimi"]
    if mimi is None:
        return
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(min(n, len(dataset))):
        item = dataset[i]
        codes = model.generate(
            item["text"].to(device),
            item["word_starts"].to(device),
            speaker=torch.tensor([item["speaker"]], device=device),
        )
        save_audio(out_dir / f"step{step:07d}_{item['id']}.wav", mimi.decode(codes))
    model.train()


def train(cfg: Config, resume: str | None = None) -> None:
    tcfg = cfg.train
    torch.manual_seed(tcfg.seed)
    random.seed(tcfg.seed)
    np.random.seed(tcfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_ds = TokenDataset(cfg.data.root, "train", cfg.data.max_utt_frames)
    val_ds = TokenDataset(cfg.data.root, "val", cfg.data.max_utt_frames)
    # The speaker table follows the data, so one config serves any prepared corpus.
    num_speakers = 1 + max(it.get("speaker", 0) for it in train_ds.items + val_ds.items)
    if num_speakers > cfg.model.num_speakers:
        print(f"model.num_speakers: {cfg.model.num_speakers} -> {num_speakers} (from the data)")
        cfg.model.num_speakers = num_speakers

    out_dir = Path(tcfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
    vocab = train_ds.vocab
    collate = make_collate(vocab.pad_id)
    sampler = FrameBudgetSampler(
        train_ds, cfg.data.max_frames_per_batch, shuffle=True, seed=tcfg.seed
    )
    train_dl = DataLoader(
        train_ds,
        batch_sampler=sampler,
        collate_fn=collate,
        num_workers=cfg.data.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.data.num_workers > 0,
    )
    val_dl = DataLoader(
        val_ds,
        batch_sampler=FrameBudgetSampler(val_ds, cfg.data.max_frames_per_batch, shuffle=False),
        collate_fn=collate,
        num_workers=0,
    )

    model = TTSModel(cfg.model, len(vocab), vocab.pad_id).to(device)
    params = count_parameters(model)
    print("parameters:", {k: f"{v / 1e6:.1f}M" for k, v in params.items()})

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": tcfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=tcfg.lr,
        betas=(0.9, 0.98),
        fused=device.type == "cuda",
    )
    autocast, scaler = autocast_for(tcfg.precision, device)

    step, best_val, epoch, epoch_step = 0, float("inf"), 0, 0
    if resume:
        ckpt = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step, best_val = ckpt["step"], ckpt.get("best_val", best_val)
        # Checkpoints without a data position come from runs that went through whole epochs.
        epoch = ckpt.get("epoch", step // max(1, len(sampler)))
        epoch_step = ckpt.get("epoch_step", step % max(1, len(sampler)))
        print(f"resumed from {resume} at step {step} (epoch {epoch}, batch {epoch_step})")

    if tcfg.compile_cfc and device.type == "cuda" and "cfc" in cfg.model.layers:
        ok = enable_compiled_step(device, autocast if autocast is not nullcontext else None)
        print(f"compiled CfC step: {'on' if ok else 'off (eager fallback)'}")
    log_f = open(out_dir / "log.jsonl", "a", encoding="utf-8")
    mimi_holder: dict = {}
    model.train()
    t0, frames_seen, running = time.time(), 0, {}

    def save_last() -> None:
        save_checkpoint(
            out_dir / "last.pt",
            model,
            optimizer,
            step,
            cfg,
            vocab,
            best_val=best_val,
            epoch=epoch,
            epoch_step=epoch_step,
        )

    # Ctrl-C / SIGTERM finish the current step, write last.pt and exit; a second one aborts.
    stop: list[int] = []

    def request_stop(signum, frame) -> None:
        if stop:
            raise KeyboardInterrupt
        stop.append(signum)
        print("\nstopping after this step (send again to abort without saving)...", flush=True)

    old_handlers = {s: signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)}

    while step < tcfg.max_steps and not stop:
        sampler.set_epoch(epoch, skip=epoch_step)
        for batch in train_dl:
            if step >= tcfg.max_steps:
                break
            for g in optimizer.param_groups:
                g["lr"] = lr_at(step, tcfg)
            batch = to_device(batch, device)
            with autocast():
                out = model(batch, tcfg)
            loss = out["loss"]
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            step += 1
            epoch_step += 1
            frames_seen += int(batch["code_lens"].sum())
            for k, v in out.items():
                running[k] = running.get(k, 0.0) + v.detach()

            if step % tcfg.log_every == 0:
                dt = time.time() - t0
                rec = {k: float(v) / tcfg.log_every for k, v in running.items()}
                rec.update(
                    step=step,
                    lr=optimizer.param_groups[0]["lr"],
                    grad_norm=float(grad_norm),
                    frames_per_s=frames_seen / dt,
                    s_per_step=dt / tcfg.log_every,
                )
                print(
                    f"step {step} loss {rec['loss']:.3f} codes {rec['loss_codes']:.3f} "
                    f"cb0 {rec['ce_cb0']:.3f} stop {rec['loss_stop']:.3f} adv {rec['loss_adv']:.3f}/{rec['adv_acc']:.2f} "
                    f"lr {rec['lr']:.2e} {rec['s_per_step']:.2f}s/step"
                )
                log_f.write(json.dumps({"split": "train", **rec}) + "\n")
                log_f.flush()
                running, frames_seen, t0 = {}, 0, time.time()

            if step % tcfg.eval_every == 0 or step == tcfg.max_steps:
                val = evaluate(model, val_dl, tcfg, device, autocast)
                print(f"[val] step {step} loss {val['loss']:.3f} codes {val['loss_codes']:.3f}")
                log_f.write(json.dumps({"split": "val", "step": step, **val}) + "\n")
                log_f.flush()
                if val["loss"] < best_val:
                    best_val = val["loss"]
                    save_checkpoint(
                        out_dir / "best.pt", model, None, step, cfg, vocab, best_val=best_val
                    )

            if step % tcfg.save_every == 0 or step == tcfg.max_steps:
                save_last()

            if tcfg.sample_every and (step % tcfg.sample_every == 0 or step == tcfg.max_steps):
                write_samples(model, val_ds, out_dir / "samples", step, device, mimi_holder)

            if stop:
                break
        else:
            epoch, epoch_step = epoch + 1, 0
    for s, handler in old_handlers.items():
        signal.signal(s, handler)
    log_f.close()
    if stop:
        save_last()
        print(f"stopped at step {step}; continue with --resume {out_dir / 'last.pt'}")
        return
    print(f"done: {step} steps, best val loss {best_val:.3f}")


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--config", required=True, help="YAML config, e.g. configs/ljspeech_cfc.yaml")
    ap.add_argument("--resume", default=None, help="path to last.pt to continue from")
    ap.add_argument(
        "overrides", nargs="*", help="e.g. train.lr=1e-4 data.max_frames_per_batch=2000"
    )


def run(args: argparse.Namespace) -> None:
    train(load_config(args.config, args.overrides), args.resume)
