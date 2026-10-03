"""The "spinal cord": a frame-rate (12.5 Hz) recurrent model that reads text through a
sliding window and speaks Mimi tokens.

Per frame the backbone sees
  * the previous frame's Mimi tokens (summed embeddings),
  * a window of characters around the current word (window cross-attention),
  * an optional speaker embedding,
and produces
  * the frame's tokens through the depth module (acoustic codebooks lag the semantic one),
  * a control decision: move the word pointer forward by 0..max_advance words,
  * an end-of-speech probability.

The text queue plus pointer is the interface a slower "cortex" model can later write into.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..config import ModelConfig, TrainConfig
from .blocks import Block, sinusoidal


class TextEncoder(nn.Module):
    """Character embeddings refined by a few local convolutions.

    Only local context is used, so text can be appended while speaking without
    changing what was already encoded (beyond the last few characters).
    """

    def __init__(self, vocab_size: int, cfg: ModelConfig, pad_id: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, cfg.text_dim, padding_idx=pad_id)
        self.convs = nn.ModuleList(
            nn.Conv1d(cfg.text_dim, cfg.text_dim, cfg.text_kernel, padding=cfg.text_kernel // 2)
            for _ in range(cfg.text_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(cfg.text_dim) for _ in range(cfg.text_layers))
        self.proj = nn.Linear(cfg.text_dim, cfg.dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, text: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        x = self.emb(text) * text_mask[..., None]
        for conv, norm in zip(self.convs, self.norms, strict=True):
            y = conv(norm(x).transpose(1, 2)).transpose(1, 2)
            x = (x + self.dropout(F.gelu(y))) * text_mask[..., None]
        return self.proj(x)


class DepthModule(nn.Module):
    """Predicts the codebooks of one row in order, conditioned on the backbone output."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        k, v, d = cfg.num_codebooks, cfg.codebook_size, cfg.depth_dim
        self.k = k
        self.inp = nn.Linear(cfg.dim, d)
        # +1: acoustic inputs can be the "not yet" token while the delay fills up
        self.code_emb = nn.ModuleList(nn.Embedding(v + 1, d) for _ in range(k - 1))
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
            nxt = sample_logits(y @ self.heads[i], temperature, top_k)
            codes = torch.cat([codes, nxt[:, None]], dim=1)
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
    text_mems: list
    text_lens: torch.Tensor
    word_starts: torch.Tensor  # [B, W]
    num_words: torch.Tensor  # [B]
    pointer: torch.Tensor  # [B] current word index
    speaker: torch.Tensor | None
    mixer_states: list
    t: int = 0


class TTSModel(nn.Module):
    def __init__(self, cfg: ModelConfig, vocab_size: int, pad_id: int):
        super().__init__()
        self.cfg = cfg
        self.pad_id = pad_id
        self.bos = cfg.codebook_size  # extra index: "no token" (start, or delayed acoustic)
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
        self.advance_head = nn.Linear(cfg.dim, cfg.max_advance + 1)
        self.depth = DepthModule(cfg)
        self.register_buffer(
            "win_offsets", torch.arange(cfg.text_window) - cfg.text_left, persistent=False
        )

    # -- codebook delay -------------------------------------------------------

    def delay(self, codes: torch.Tensor) -> torch.Tensor:
        """``[B, T, K]`` frames -> ``[B, T + d, K]`` rows: row s holds the semantic token of
        frame s and the acoustic tokens of frame s - d ("no token" where undefined)."""
        d = self.cfg.acoustic_delay
        b, t, k = codes.shape
        grid = torch.full((b, t + d, k), self.bos, dtype=codes.dtype, device=codes.device)
        grid[:, :t, 0] = codes[:, :, 0]
        grid[:, d:, 1:] = codes[:, :, 1:]
        return grid

    def undelay(self, rows: torch.Tensor) -> torch.Tensor:
        """``[S, K]`` rows -> ``[S - d, K]`` frames."""
        d = self.cfg.acoustic_delay
        frames = rows[: rows.shape[0] - d].clone()
        frames[:, 1:] = rows[d:, 1:]
        return frames

    # -- text window ----------------------------------------------------------

    def window(self, word_starts, pointer, text_lens):
        """Window positions / validity ``[..., W]`` for word ``pointer`` (``[B]`` or ``[B, S]``)."""
        flat = pointer if pointer.dim() == 2 else pointer[:, None]
        start = word_starts.gather(1, flat)
        idx = start[..., None] + self.win_offsets
        lens = text_lens.view(-1, *([1] * (idx.dim() - 1)))
        mask = (idx >= 0) & (idx < lens)
        idx = idx.clamp(min=0) % lens.clamp(min=1)  # any valid position; masked anyway
        if pointer.dim() == 1:
            return idx[:, 0], mask[:, 0]
        return idx, mask

    def pointers(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher pointer (current word per row) and advance targets, both ``[B, S]``."""
        d = self.cfg.acoustic_delay
        code_lens, word_frames, num_words = (
            batch["code_lens"],
            batch["word_frames"],
            batch["num_words"],
        )
        s = batch["codes"].shape[1] + d
        rows = torch.arange(s, device=code_lens.device)
        frame = torch.minimum(rows[None], code_lens[:, None] - 1)  # [B, S]
        started = (word_frames[:, None, :] <= frame[..., None]).sum(-1) - 1
        pointer = torch.minimum(started.clamp(min=0), num_words[:, None] - 1)
        adv = torch.zeros_like(pointer)
        adv[:, :-1] = (pointer[:, 1:] - pointer[:, :-1]).clamp(0, self.cfg.max_advance)
        return pointer, adv

    # -- shared pieces --------------------------------------------------------

    def _embed_rows(self, rows: torch.Tensor) -> torch.Tensor:
        return sum(emb(rows[..., i]) for i, emb in enumerate(self.audio_emb))

    def _frame_input(self, x, positions, speaker):
        if self.cfg.audio_pos_emb:
            x = x + sinusoidal(positions, self.cfg.dim)
        if self.speaker_emb is not None and speaker is not None:
            s = self.speaker_emb(speaker)
            x = x + (s.unsqueeze(1) if x.dim() == 3 else s)
        return x

    def encode_text(self, text: torch.Tensor, text_lens: torch.Tensor) -> list:
        mask = torch.arange(text.shape[1], device=text.device)[None] < text_lens[:, None]
        enc = self.text_encoder(text, mask)
        return [blk.cross.memory(enc) if blk.cross is not None else None for blk in self.blocks]

    # -- training -------------------------------------------------------------

    def backbone(self, batch: dict, rows: torch.Tensor, pointer: torch.Tensor) -> torch.Tensor:
        b, s, k = rows.shape
        mems = self.encode_text(batch["text"], batch["text_lens"])
        win_idx, win_mask = self.window(batch["word_starts"], pointer, batch["text_lens"])
        start = torch.full((b, 1, k), self.bos, dtype=rows.dtype, device=rows.device)
        prev = torch.cat([start, rows[:, :-1]], dim=1)
        x = self._frame_input(
            self._embed_rows(prev), torch.arange(s, device=rows.device), batch.get("speaker")
        )
        for blk, mem in zip(self.blocks, mems, strict=True):
            x = blk(x, mem, win_idx, win_mask)
        return self.norm(x)

    def forward(self, batch: dict, tcfg: TrainConfig) -> dict[str, torch.Tensor]:
        d, k = self.cfg.acoustic_delay, self.cfg.num_codebooks
        rows = self.delay(batch["codes"])
        b, s, _ = rows.shape
        row_lens = batch["code_lens"] + d
        r = torch.arange(s, device=rows.device)[None]
        row_mask = r < row_lens[:, None]  # [B, S]
        # Which (row, codebook) entries are real tokens.
        tok_mask = torch.zeros(b, s, k, dtype=torch.bool, device=rows.device)
        tok_mask[..., 0] = r < batch["code_lens"][:, None]
        tok_mask[..., 1:] = ((r >= d) & row_mask)[..., None]

        pointer, adv = self.pointers(batch)
        h = self.backbone(batch, rows, pointer)

        logits = self.depth(h[row_mask], rows[row_mask])  # [N, K, V]
        target = rows[row_mask].masked_fill(~tok_mask[row_mask], -100)
        ce = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            target.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).view(-1, k)
        valid = tok_mask[row_mask].float()
        ce = (ce * valid).sum(0) / valid.sum(0).clamp(min=1)
        w = torch.tensor(tcfg.codebook_weights[:k], device=ce.device, dtype=ce.dtype)
        loss_codes = (ce * w).sum() / w.sum()

        stop_logit = self.stop_head(h).squeeze(-1).float()
        stop_target = (r == (row_lens[:, None] - 1)).float()
        loss_stop = F.binary_cross_entropy_with_logits(
            stop_logit[row_mask],
            stop_target[row_mask],
            pos_weight=torch.tensor(20.0, device=h.device),
        )

        adv_logits = self.advance_head(h).float()
        loss_adv = F.cross_entropy(adv_logits[row_mask], adv[row_mask])
        adv_acc = (adv_logits[row_mask].argmax(-1) == adv[row_mask]).float().mean()

        loss = loss_codes + tcfg.stop_weight * loss_stop + tcfg.advance_weight * loss_adv
        out = {
            "loss": loss,
            "loss_codes": loss_codes,
            "loss_stop": loss_stop,
            "loss_adv": loss_adv,
            "adv_acc": adv_acc,
        }
        for i in range(k):
            out[f"ce_cb{i}"] = ce[i].detach()
        return out

    # -- streaming inference --------------------------------------------------

    def start(self, text, text_lens, word_starts, num_words, speaker=None) -> StreamState:
        mems = self.encode_text(text, text_lens)
        b, device = text.shape[0], text.device
        states = [blk.mixer.init_state(b, device) for blk in self.blocks]
        pointer = torch.zeros(b, dtype=torch.long, device=device)
        return StreamState(mems, text_lens, word_starts, num_words, pointer, speaker, states, 0)

    def step(self, prev_row: torch.Tensor | None, state: StreamState):
        """Advance one row. ``prev_row`` ``[B, K]`` (None for the first row).
        Uses ``state.pointer`` as the current word; the caller moves it.
        Returns backbone output ``[B, D]``, stop probability ``[B]``, advance logits ``[B, A]``."""
        b = state.pointer.shape[0]
        device = state.pointer.device
        if prev_row is None:
            prev_row = torch.full(
                (b, self.cfg.num_codebooks), self.bos, dtype=torch.long, device=device
            )
        pos = torch.full((b,), state.t, dtype=torch.long, device=device)
        x = self._frame_input(self._embed_rows(prev_row), pos, state.speaker)
        win_idx, win_mask = self.window(state.word_starts, state.pointer, state.text_lens)
        for i, (blk, mem) in enumerate(zip(self.blocks, state.text_mems, strict=True)):
            x, state.mixer_states[i] = blk.step(x, mem, win_idx, win_mask, state.mixer_states[i])
        state.t += 1
        h = self.norm(x)
        return h, torch.sigmoid(self.stop_head(h).squeeze(-1)), self.advance_head(h)

    @torch.no_grad()
    def generate(
        self,
        text: torch.Tensor,
        word_starts: torch.Tensor,
        speaker: torch.Tensor | None = None,
        max_frames: int = 1000,
        temperature: float = 0.8,
        top_k: int = 50,
        stop_threshold: float = 0.5,
        max_frames_per_word: int = 40,
        on_row=None,
    ) -> torch.Tensor:
        """Generate frames ``[T, K]`` for one utterance: ``text`` ``[N]``, ``word_starts`` ``[W]``.

        Guards: speech may only end on the last word, and the pointer is forced forward
        after ``max_frames_per_word`` frames on one word. ``on_row(s, row)`` is called
        for every generated row (before undelaying) for streaming consumers.
        """
        d = self.cfg.acoustic_delay
        device = text.device
        n_words = len(word_starts)
        state = self.start(
            text[None],
            torch.tensor([len(text)], device=device),
            word_starts[None],
            torch.tensor([n_words], device=device),
            speaker,
        )
        prev, rows, on_word = None, [], 0
        for s in range(max_frames + d):
            h, stop, adv_logits = self.step(prev, state)
            row = self.depth.sample(h, temperature, top_k)
            if s < d:
                row[:, 1:] = self.bos
            rows.append(row[0])
            if on_row is not None:
                on_row(s, row[0])
            last_word = int(state.pointer) == n_words - 1
            if s + 1 > d and last_word and stop.item() > stop_threshold:
                break
            adv = int(adv_logits.argmax(-1))
            on_word = 0 if adv else on_word + 1
            if on_word >= max_frames_per_word:
                adv, on_word = 1, 0
            state.pointer = (state.pointer + adv).clamp(max=n_words - 1)
            prev = row
        return self.undelay(torch.stack(rows))


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
