"""Experiment configuration: dataclasses loaded from YAML."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    root: str = "data/ljspeech_tokens"  # output of `aqyn prepare`
    max_frames_per_batch: int = 3000  # batch size in Mimi frames (12.5 per second)
    max_utt_frames: int = 250  # drop longer utterances (20 s)
    num_workers: int = 2


@dataclass
class ModelConfig:
    # Mimi
    num_codebooks: int = 8
    codebook_size: int = 2048
    # Text: character embeddings + local convolutions (no look at the whole text)
    text_dim: int = 384
    text_layers: int = 3
    text_kernel: int = 5
    # Sliding text window read by the backbone at every frame
    text_window: int = 32  # characters visible per frame
    text_left: int = 8  # of which this many are before the current word
    max_advance: int = 3  # the control head moves the pointer by 0..max_advance words per frame
    # Acoustic codebooks lag the semantic one by this many frames (Moshi-style)
    acoustic_delay: int = 1
    # Temporal backbone
    dim: int = 512
    # One entry per block: "cfc", "lstm", "attn" (full causal) or "local" (sliding window)
    layers: list[str] = field(default_factory=lambda: ["cfc"] * 12)
    heads: int = 8
    ffn_mult: int = 4
    local_window: int = 100  # frames, 100 = 8 s
    audio_pos_emb: bool = False  # sinusoidal positions on audio input (needed for attention-only)
    cross_attn_every: int = 4  # window cross-attention in every n-th block
    # CfC cell
    cfc_mode: str = "default"  # "default", "no_gate" or "pure"
    cfc_backbone_units: int = 512
    cfc_backbone_layers: int = 1
    # Depth module (codebooks within a frame)
    depth_dim: int = 256
    depth_layers: int = 4
    depth_heads: int = 4
    # Conditioning
    num_speakers: int = 1
    dropout: float = 0.1


@dataclass
class TrainConfig:
    out_dir: str = "runs/default"
    max_steps: int = 100_000
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 2000
    grad_clip: float = 1.0
    precision: str = "bf16"  # "bf16", "fp16" or "fp32"
    log_every: int = 50
    eval_every: int = 2000
    save_every: int = 2000
    sample_every: int = 10_000  # synthesize audio samples (needs Mimi); 0 to disable
    codebook_weights: list[float] = field(default_factory=lambda: [1.0] * 8)
    stop_weight: float = 1.0
    advance_weight: float = 1.0
    seed: int = 0
    compile_cfc: bool = True  # fuse the CfC step with torch.compile on CUDA (falls back to eager)


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _merge(dc_type: type, values: dict[str, Any] | None):
    values = values or {}
    known = {f.name for f in dataclasses.fields(dc_type)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown keys for {dc_type.__name__}: {sorted(unknown)}")
    return dc_type(**values)


def config_from_dict(d: dict[str, Any]) -> Config:
    unknown = set(d) - {"data", "model", "train"}
    if unknown:
        raise ValueError(f"Unknown config sections: {sorted(unknown)}")
    return Config(
        data=_merge(DataConfig, d.get("data")),
        model=_merge(ModelConfig, d.get("model")),
        train=_merge(TrainConfig, d.get("train")),
    )


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load a YAML config. Overrides look like ``train.lr=1e-4``."""
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    for item in overrides or []:
        key, _, raw = item.partition("=")
        section, _, name = key.partition(".")
        if not name:
            raise ValueError(f"Override must look like section.key=value, got {item!r}")
        d.setdefault(section, {})[name] = yaml.safe_load(raw)
    return config_from_dict(d)
