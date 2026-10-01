"""Backbone building blocks. Each has a parallel ``forward`` for training and a
``step`` for frame-by-frame streaming that gives the same result."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .cfc import CfC


def sinusoidal(positions: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embeddings for integer ``positions`` of any shape -> ``[..., dim]``."""
    half = dim // 2
    freqs = torch.exp(-math.log(10_000.0) * torch.arange(half, device=positions.device) / half)
    angles = positions.float().unsqueeze(-1) * freqs
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


class FFN(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.up = nn.Linear(dim, dim * mult)
        self.down = nn.Linear(dim * mult, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.dropout(F.gelu(self.up(x))))


# ---------------------------------------------------------------------------
# Mixers: process the audio-frame sequence causally.
# ---------------------------------------------------------------------------


class CfCMixer(nn.Module):
    def __init__(
        self, dim: int, backbone_units: int, backbone_layers: int, mode: str, dropout: float
    ):
        super().__init__()
        self.cfc = CfC(
            dim,
            dim,
            backbone_units=backbone_units,
            backbone_layers=backbone_layers,
            mode=mode,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cfc(x)[0]

    def init_state(self, batch: int, device: torch.device):
        return self.cfc.initial_state(batch).to(device)

    def step(self, x: torch.Tensor, state):
        h = self.cfc.step(x, state)
        return h, h


class LSTMMixer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lstm = nn.LSTM(dim, dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lstm(x)[0]

    def init_state(self, batch: int, device: torch.device):
        z = torch.zeros(1, batch, self.lstm.hidden_size, device=device)
        return (z, z.clone())

    def step(self, x: torch.Tensor, state):
        y, state = self.lstm(x.unsqueeze(1), state)
        return y[:, 0], state


class SelfAttnMixer(nn.Module):
    """Causal self-attention, optionally restricted to the last ``window`` frames."""

    def __init__(self, dim: int, heads: int, window: int | None = None, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.window = window
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = dropout

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        return x.view(b, t, self.heads, d // self.heads).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = (self._split(z) for z in self.qkv(x).chunk(3, dim=-1))
        i = torch.arange(t, device=x.device)
        allowed = i[None, :] <= i[:, None]
        if self.window is not None:
            allowed &= i[None, :] > i[:, None] - self.window
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed, dropout_p=self.dropout if self.training else 0.0
        )
        return self.out(y.transpose(1, 2).reshape(b, t, d))

    def init_state(self, batch: int, device: torch.device):
        return None  # KV cache created on the first step

    def step(self, x: torch.Tensor, state):
        b, d = x.shape
        q, k, v = (self._split(z) for z in self.qkv(x.unsqueeze(1)).chunk(3, dim=-1))
        if state is not None:
            k = torch.cat([state[0], k], dim=2)
            v = torch.cat([state[1], v], dim=2)
        if self.window is not None:
            k, v = k[:, :, -self.window :], v[:, :, -self.window :]
        y = F.scaled_dot_product_attention(q, k, v)
        return self.out(y.transpose(1, 2).reshape(b, d)), (k, v)


# ---------------------------------------------------------------------------
# Cross-attention to the encoded text.
# ---------------------------------------------------------------------------


class CrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def memory(self, text: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Precompute keys/values of the text once: ``[B, H, N, dh]`` each."""
        b, n, d = text.shape
        k, v = self.kv(text).chunk(2, dim=-1)
        k = k.view(b, n, self.heads, d // self.heads).transpose(1, 2)
        v = v.view(b, n, self.heads, d // self.heads).transpose(1, 2)
        return k, v

    def forward(
        self, x: torch.Tensor, mem: tuple[torch.Tensor, torch.Tensor], text_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``x`` ``[B, T, D]``; ``text_mask`` ``[B, N]`` True for real tokens.
        Returns the output and head-averaged attention weights ``[B, T, N]``."""
        b, t, d = x.shape
        k, v = mem
        q = self.q(x).view(b, t, self.heads, d // self.heads).transpose(1, 2)
        scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
        scores = scores.masked_fill(~text_mask[:, None, None, :], float("-inf"))
        attn = scores.softmax(dim=-1)
        y = self.dropout(attn) @ v
        y = self.out(y.transpose(1, 2).reshape(b, t, d))
        return y, attn.mean(dim=1)


# ---------------------------------------------------------------------------
# Backbone block.
# ---------------------------------------------------------------------------


def make_mixer(kind: str, cfg) -> nn.Module:
    if kind == "cfc":
        return CfCMixer(
            cfg.dim, cfg.cfc_backbone_units, cfg.cfc_backbone_layers, cfg.cfc_mode, cfg.dropout
        )
    if kind == "lstm":
        return LSTMMixer(cfg.dim)
    if kind == "attn":
        return SelfAttnMixer(cfg.dim, cfg.heads, None, cfg.dropout)
    if kind == "local":
        return SelfAttnMixer(cfg.dim, cfg.heads, cfg.local_window, cfg.dropout)
    raise ValueError(f"Unknown mixer {kind!r}; expected cfc, lstm, attn or local")


class Block(nn.Module):
    def __init__(self, kind: str, cfg, cross: bool):
        super().__init__()
        self.kind = kind
        self.norm_mix = nn.LayerNorm(cfg.dim)
        self.mixer = make_mixer(kind, cfg)
        self.cross = CrossAttention(cfg.dim, cfg.heads, cfg.dropout) if cross else None
        self.norm_cross = nn.LayerNorm(cfg.dim) if cross else None
        self.norm_ffn = nn.LayerNorm(cfg.dim)
        self.ffn = FFN(cfg.dim, cfg.ffn_mult, cfg.dropout)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, mem, text_mask):
        x = x + self.dropout(self.mixer(self.norm_mix(x)))
        attn = None
        if self.cross is not None:
            y, attn = self.cross(self.norm_cross(x), mem, text_mask)
            x = x + self.dropout(y)
        x = x + self.dropout(self.ffn(self.norm_ffn(x)))
        return x, attn

    def step(self, x, mem, text_mask, state):
        y, state = self.mixer.step(self.norm_mix(x), state)
        x = x + y
        attn = None
        if self.cross is not None:
            y, attn = self.cross(self.norm_cross(x).unsqueeze(1), mem, text_mask)
            x = x + y[:, 0]
            attn = attn[:, 0]
        x = x + self.ffn(self.norm_ffn(x))
        return x, state, attn
