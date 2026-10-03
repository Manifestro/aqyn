"""Text-to-speech model: text encoder -> recurrent backbone over Mimi frames -> depth module."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..config import ModelConfig, TrainConfig
from .blocks import Block, sinusoidal


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, cfg: ModelConfig, pad_id: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, cfg.text_dim, padding_idx=pad_id)
        layer = nn.TransformerEncoderLayer(
            cfg.text_dim,
            cfg.text_heads,
            dim_feedforward=cfg.text_dim * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, cfg.text_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.text_dim)
        self.proj = nn.Linear(cfg.text_dim, cfg.dim)

    def forward(self, text: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(text.shape[1], device=text.device)
        x = self.emb(text) * math.sqrt(self.emb.embedding_dim) + sinusoidal(
            pos, self.emb.embedding_dim
        )
        x = self.encoder(x, src_key_padding_mask=~text_mask)
        return self.proj(self.norm(x))


class DepthModule(nn.Module):
    """Predicts the codebooks of one frame in order, conditioned on the backbone output."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        k, v, d = cfg.num_codebooks, cfg.codebook_size, cfg.depth_dim
        self.k = k
        self.inp = nn.Linear(cfg.dim, d)
        self.code_emb = nn.ModuleList(nn.Embedding(v, d) for _ in range(k - 1))
        self.pos = nn.Parameter(torch.zeros(k, d))
        layer = nn.TransformerEncoderLayer(
            d,
            cfg.depth_heads,
            d * 4,
            cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.tf = nn.TransformerEncoder(layer, cfg.depth_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.heads = nn.Parameter(torch.empty(k, d, v))
        nn.init.normal_(self.heads, std=d**-0.5)
        nn.init.normal_(self.pos, std=0.02)
        self.register_buffer(
            "causal", torch.triu(torch.ones(k, k, dtype=torch.bool), 1), persistent=False
        )

    def _inputs(self, h: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """``h`` ``[N, D]``, ``codes`` ``[N, j]`` with j < K -> inputs ``[N, j+1, d]``."""
        x = [self.inp(h)]
        for i in range(codes.shape[1]):
            x.append(self.code_emb[i](codes[:, i]))
        x = torch.stack(x, dim=1)
        return x + self.pos[: x.shape[1]]

    def forward(self, h: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """Teacher-forced logits ``[N, K, V]``."""
        x = self._inputs(h, codes[:, :-1])
        y = self.norm(self.tf(x, mask=self.causal, is_causal=True))
        return torch.einsum("nkd,kdv->nkv", y, self.heads)

    @torch.no_grad()
    def sample(self, h: torch.Tensor, temperature: float = 0.8, top_k: int = 50) -> torch.Tensor:
        codes = torch.zeros(h.shape[0], 0, dtype=torch.long, device=h.device)
        for i in range(self.k):
            x = self._inputs(h, codes)
            n = x.shape[1]
            y = self.norm(self.tf(x, mask=self.causal[:n, :n], is_causal=True))[:, -1]
            logits = y @ self.heads[i]
            codes = torch.cat([codes, sample_logits(logits, temperature, top_k)[:, None]], dim=1)
        return codes


def sample_logits(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1)
    logits = logits.float() / temperature
    if top_k > 0:
        kth = logits.topk(min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    return torch.multinomial(logits.softmax(dim=-1), 1)[:, 0]


@dataclass
class StreamState:
    text_mem: list
    text_mask: torch.Tensor
    speaker: torch.Tensor | None
    mixer_states: list
    t: int = 0


class TTSModel(nn.Module):
    def __init__(self, cfg: ModelConfig, vocab_size: int, pad_id: int):
        super().__init__()
        self.cfg = cfg
        self.pad_id = pad_id
        self.bos = cfg.codebook_size  # extra index in each audio embedding table
        self.text_encoder = TextEncoder(vocab_size, cfg, pad_id)
        self.audio_emb = nn.ModuleList(
            nn.Embedding(cfg.codebook_size + 1, cfg.dim) for _ in range(cfg.num_codebooks)
        )
        self.speaker_emb = nn.Embedding(cfg.num_speakers, cfg.dim) if cfg.num_speakers > 1 else None
        self.blocks = nn.ModuleList(
            Block(kind, cfg, cross=(i % cfg.cross_attn_every == 0))
            for i, kind in enumerate(cfg.layers)
        )
        self.norm = nn.LayerNorm(cfg.dim)
        self.stop_head = nn.Linear(cfg.dim, 1)
        self.depth = DepthModule(cfg)

    # -- shared pieces ------------------------------------------------------

    def _embed_frames(self, codes: torch.Tensor) -> torch.Tensor:
        """``codes`` ``[..., K]`` -> summed embeddings ``[..., D]``."""
        return sum(emb(codes[..., i]) for i, emb in enumerate(self.audio_emb))

    def _frame_input(self, x: torch.Tensor, positions: torch.Tensor, speaker: torch.Tensor | None):
        if self.cfg.audio_pos_emb:
            x = x + sinusoidal(positions, self.cfg.dim)
        if self.speaker_emb is not None and speaker is not None:
            s = self.speaker_emb(speaker)
            x = x + (s.unsqueeze(1) if x.dim() == 3 else s)
        return x

    def encode_text(self, text: torch.Tensor, text_lens: torch.Tensor):
        mask = torch.arange(text.shape[1], device=text.device)[None] < text_lens[:, None]
        enc = self.text_encoder(text, mask)
        mems = [blk.cross.memory(enc) if blk.cross is not None else None for blk in self.blocks]
        return mems, mask

    # -- training -----------------------------------------------------------

    def backbone(self, batch: dict) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        codes = batch["codes"]
        b, t, k = codes.shape
        mems, text_mask = self.encode_text(batch["text"], batch["text_lens"])
        bos = torch.full((b, 1, k), self.bos, dtype=codes.dtype, device=codes.device)
        prev = torch.cat([bos, codes[:, :-1]], dim=1)
        x = self._frame_input(
            self._embed_frames(prev), torch.arange(t, device=codes.device), batch.get("speaker")
        )
        attns = []
        for blk, mem in zip(self.blocks, mems, strict=True):
            x, attn = blk(x, mem, text_mask)
            if attn is not None:
                attns.append(attn)
        return self.norm(x), attns, text_mask

    def forward(self, batch: dict, tcfg: TrainConfig) -> dict[str, torch.Tensor]:
        codes, code_lens = batch["codes"], batch["code_lens"]
        _, t, k = codes.shape
        h, attns, text_mask = self.backbone(batch)
        frame_mask = torch.arange(t, device=codes.device)[None] < code_lens[:, None]

        # Codebook cross-entropy on real frames only.
        logits = self.depth(h[frame_mask], codes[frame_mask])  # [N, K, V]
        target = codes[frame_mask]
        ce = (
            F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none"
            )
            .view(-1, k)
            .mean(dim=0)
        )
        w = torch.tensor(tcfg.codebook_weights[:k], device=ce.device, dtype=ce.dtype)
        loss_codes = (ce * w).sum() / w.sum()

        # Stop prediction: positive on the last real frame.
        stop_logit = self.stop_head(h).squeeze(-1).float()
        stop_target = (
            torch.arange(t, device=codes.device)[None] == (code_lens[:, None] - 1)
        ).float()
        loss_stop = F.binary_cross_entropy_with_logits(
            stop_logit[frame_mask],
            stop_target[frame_mask],
            pos_weight=torch.tensor(20.0, device=codes.device),
        )

        loss_guide = guided_attention_loss(attns, frame_mask, text_mask, tcfg.guided_attn_sigma)
        loss = loss_codes + tcfg.stop_weight * loss_stop + tcfg.guided_attn_weight * loss_guide
        out = {
            "loss": loss,
            "loss_codes": loss_codes,
            "loss_stop": loss_stop,
            "loss_guide": loss_guide,
        }
        for i in range(k):
            out[f"ce_cb{i}"] = ce[i].detach()
        return out

    # -- streaming inference ------------------------------------------------

    def start(
        self, text: torch.Tensor, text_lens: torch.Tensor, speaker: torch.Tensor | None = None
    ):
        mems, text_mask = self.encode_text(text, text_lens)
        b, device = text.shape[0], text.device
        states = [blk.mixer.init_state(b, device) for blk in self.blocks]
        return StreamState(mems, text_mask, speaker, states, 0)

    def step(self, prev_codes: torch.Tensor | None, state: StreamState):
        """Advance one frame. ``prev_codes`` ``[B, K]`` (None for the first frame).
        Returns backbone output ``[B, D]``, stop probability ``[B]``, cross-attention ``[B, N]``."""
        b = state.text_mask.shape[0]
        if prev_codes is None:
            prev_codes = torch.full(
                (b, self.cfg.num_codebooks),
                self.bos,
                dtype=torch.long,
                device=state.text_mask.device,
            )
        pos = torch.full((b,), state.t, dtype=torch.long, device=prev_codes.device)
        x = self._frame_input(self._embed_frames(prev_codes), pos, state.speaker)
        attns = []
        for i, (blk, mem) in enumerate(zip(self.blocks, state.text_mem, strict=True)):
            x, state.mixer_states[i], attn = blk.step(
                x, mem, state.text_mask, state.mixer_states[i]
            )
            if attn is not None:
                attns.append(attn)
        state.t += 1
        h = self.norm(x)
        stop = torch.sigmoid(self.stop_head(h).squeeze(-1))
        attn = torch.stack(attns).mean(0) if attns else None
        return h, stop, attn

    @torch.no_grad()
    def generate(
        self,
        text: torch.Tensor,
        speaker: torch.Tensor | None = None,
        max_frames: int = 500,
        min_frames: int = 5,
        temperature: float = 0.8,
        top_k: int = 50,
        stop_threshold: float = 0.5,
        on_frame=None,
    ) -> torch.Tensor:
        """Generate codes ``[T, K]`` for a single utterance ``text`` ``[N]``.
        ``on_frame(t, codes)`` is called after each frame, for streaming consumers."""
        text = text.unsqueeze(0)
        state = self.start(text, torch.tensor([text.shape[1]], device=text.device), speaker)
        prev, frames = None, []
        for t in range(max_frames):
            h, stop, _ = self.step(prev, state)
            prev = self.depth.sample(h, temperature, top_k)
            frames.append(prev[0])
            if on_frame is not None:
                on_frame(t, prev[0])
            if t + 1 >= min_frames and stop.item() > stop_threshold:
                break
        return torch.stack(frames)


def guided_attention_loss(
    attns: list[torch.Tensor], frame_mask: torch.Tensor, text_mask: torch.Tensor, sigma: float
) -> torch.Tensor:
    """Penalize attention far from the diagonal (Tachibana et al., 2018)."""
    if not attns:
        return frame_mask.new_zeros((), dtype=torch.float32)
    a = torch.stack(attns).mean(0).float()  # [B, T, N]
    t_len = frame_mask.sum(1).clamp(min=1).float()
    n_len = text_mask.sum(1).clamp(min=1).float()
    t_pos = torch.arange(a.shape[1], device=a.device)[None, :, None] / t_len[:, None, None]
    n_pos = torch.arange(a.shape[2], device=a.device)[None, None, :] / n_len[:, None, None]
    w = 1.0 - torch.exp(-((n_pos - t_pos) ** 2) / (2 * sigma**2))
    mask = frame_mask[:, :, None] & text_mask[:, None, :]
    return (a * w)[mask].sum() / frame_mask.sum().clamp(min=1)


def count_parameters(model: nn.Module) -> dict[str, int]:
    def n(m):
        return sum(p.numel() for p in m.parameters())

    return {
        "total": n(model),
        "text_encoder": n(model.text_encoder),
        "audio_emb": n(model.audio_emb),
        "backbone": n(model.blocks),
        "depth": n(model.depth),
    }
