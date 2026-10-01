"""Saving and loading checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch

from .config import Config, config_from_dict
from .models import TTSModel
from .text import CharVocab


def save_checkpoint(
    path: str | Path, model, optimizer, step: int, cfg: Config, vocab: CharVocab, **extra
):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "step": step,
            "config": cfg.to_dict(),
            "vocab": vocab.symbols,
            **extra,
        },
        tmp,
    )
    tmp.replace(path)


def load_model(path: str | Path, device: str | torch.device = "cpu"):
    """Returns ``(model, vocab, cfg, checkpoint_dict)`` with the model in eval mode."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = config_from_dict(ckpt["config"])
    vocab = CharVocab(ckpt["vocab"])
    model = TTSModel(cfg.model, len(vocab), vocab.pad_id)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), vocab, cfg, ckpt
