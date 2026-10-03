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
# Cross-attention to a sliding window of the text.
# ---------------------------------------------------------------------------


class WindowCrossAttention(nn.Module):
    """Each frame attends only to ``window`` text positions around its current word.

    Keys and values are computed once for the whole text and gathered per frame, so the
    per-frame cost is fixed no matter how long the text is. A learned bias per head and
    window offset tells the model where each character sits relative to the current word.
    """

    def __init__(self, dim: int, heads: int, window: int, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.out = nn.Linear(dim, dim)
        self.rel_bias = nn.Parameter(torch.zeros(heads, window))
        self.dropout = nn.Dropout(dropout)

    def memory(self, text: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Keys/values for the whole text: ``[B, N, D]`` each."""
        k, v = self.kv(text).chunk(2, dim=-1)
        return k, v

    def forward(
        self,
        x: torch.Tensor,
        mem: tuple[torch.Tensor, torch.Tensor],
        win_idx: torch.Tensor,
        win_mask: torch.Tensor,
    ) -> torch.Tensor:
        """``x`` ``[B, S, D]``; ``win_idx`` / ``win_mask`` ``[B, S, W]`` (text positions, validity)."""
        b, s, d = x.shape
        h, dh, w = self.heads, d // self.heads, win_idx.shape[-1]
        k, v = mem
        flat = win_idx.reshape(b, s * w, 1).expand(-1, -1, d)
        k = k.gather(1, flat).view(b, s, w, h, dh)
        v = v.gather(1, flat).view(b, s, w, h, dh)
        q = self.q(x).view(b, s, h, dh)
        scores = torch.einsum("bshd,bswhd->bshw", q, k) / math.sqrt(dh)
        scores = scores + self.rel_bias.to(scores.dtype)
        scores = scores.masked_fill(~win_mask[:, :, None, :], float("-inf"))
        attn = self.dropout(scores.softmax(dim=-1))
        y = torch.einsum("bshw,bswhd->bshd", attn, v).reshape(b, s, d)
        return self.out(y)


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
        self.cross = (
            WindowCrossAttention(cfg.dim, cfg.heads, cfg.text_window, cfg.dropout)
            if cross
            else None
        )
        self.norm_cross = nn.LayerNorm(cfg.dim) if cross else None
        self.norm_ffn = nn.LayerNorm(cfg.dim)
        self.ffn = FFN(cfg.dim, cfg.ffn_mult, cfg.dropout)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, mem, win_idx, win_mask):
        x = x + self.dropout(self.mixer(self.norm_mix(x)))
        if self.cross is not None:
            x = x + self.dropout(self.cross(self.norm_cross(x), mem, win_idx, win_mask))
        return x + self.dropout(self.ffn(self.norm_ffn(x)))

    def step(self, x, mem, win_idx, win_mask, state):
        """One frame: ``x`` ``[B, D]``, ``win_idx`` / ``win_mask`` ``[B, W]``."""
        y, state = self.mixer.step(self.norm_mix(x), state)
        x = x + y
        if self.cross is not None:
            y = self.cross(
                self.norm_cross(x).unsqueeze(1), mem, win_idx[:, None], win_mask[:, None]
            )
            x = x + y[:, 0]
        return x + self.ffn(self.norm_ffn(x)), state
