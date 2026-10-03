"""Word-level forced alignment with a CTC speech model.

Uses any character-level CTC model from Hugging Face transformers (English default:
``facebook/wav2vec2-base-960h``; for Russian and Kazakh, MMS models can be swapped in)
and a Viterbi pass over the known transcript. Returns the start and end time of every word.
"""

from __future__ import annotations

import re

import numpy as np
import soxr
import torch

CTC_SAMPLE_RATE = 16_000


def ctc_viterbi(log_probs: np.ndarray, tokens: list[int], blank: int) -> np.ndarray | None:
    """Best CTC path of ``tokens`` through ``log_probs`` ``[T, V]``.

    Returns, for every emission frame, the index of the token it belongs to,
    or -1 for blank frames. None if the audio is too short for the transcript.
    """
    t_len, n = log_probs.shape[0], len(tokens)
    ext = np.full(2 * n + 1, blank, dtype=np.int64)
    ext[1::2] = tokens
    s_len = len(ext)
    # A label may skip the blank before it unless it repeats the previous label.
    can_skip = np.zeros(s_len, dtype=bool)
    can_skip[3::2] = ext[3::2] != ext[1:-2:2]

    neg = -np.inf
    dp = np.full(s_len, neg)
    dp[0] = log_probs[0, ext[0]]
    if s_len > 1:
        dp[1] = log_probs[0, ext[1]]
    back = np.zeros((t_len, s_len), dtype=np.int8)
    for t in range(1, t_len):
        stay = dp
        prev1 = np.concatenate([[neg], dp[:-1]])
        prev2 = np.where(can_skip, np.concatenate([[neg, neg], dp[:-2]]), neg)
        cand = np.stack([stay, prev1, prev2])
        choice = cand.argmax(axis=0)
        dp = cand[choice, np.arange(s_len)] + log_probs[t, ext]
        back[t] = choice

    ends = [s_len - 1, s_len - 2] if s_len > 1 else [0]
    s = max(ends, key=lambda i: dp[i])
    if not np.isfinite(dp[s]):
        return None
    path = np.empty(t_len, dtype=np.int64)
    for t in range(t_len - 1, -1, -1):
        path[t] = s
        s -= int(back[t, s])
    return np.where(path % 2 == 1, (path - 1) // 2, -1)


class CTCAligner:
    def __init__(self, model_id: str = "facebook/wav2vec2-base-960h", device: str = "cpu"):
        from transformers import AutoModelForCTC, AutoProcessor

        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForCTC.from_pretrained(model_id).to(self.device).eval()
        vocab = self.processor.tokenizer.get_vocab()
        self.blank = self.processor.tokenizer.pad_token_id
        self.delim = vocab.get(self.processor.tokenizer.word_delimiter_token)
        self.upper = any(c.isupper() for c in vocab if len(c) == 1)
        self.char_ids = {c: i for c, i in vocab.items() if len(c) == 1}
        # Seconds per emission frame (wav2vec2 / MMS: 320 samples at 16 kHz = 20 ms).
        self.stride = float(np.prod(self.model.config.conv_stride)) / CTC_SAMPLE_RATE

    def _word_tokens(self, word: str) -> list[int]:
        w = word.upper() if self.upper else word.lower()
        return [self.char_ids[c] for c in re.sub(r"\+", "", w) if c in self.char_ids]

    @torch.inference_mode()
    def align(self, wav: np.ndarray, sr: int, words: list[str]) -> list[tuple[float, float]] | None:
        """Start/end seconds for each word. Words with no alignable characters take
        the timing of their neighbour. None if alignment fails."""
        if sr != CTC_SAMPLE_RATE:
            wav = soxr.resample(wav, sr, CTC_SAMPLE_RATE)
        feats = self.processor(wav, sampling_rate=CTC_SAMPLE_RATE, return_tensors="pt")
        logits = self.model(feats.input_values.to(self.device)).logits[0]
        log_probs = logits.float().log_softmax(-1).cpu().numpy()

        tokens, owner = [], []  # owner[i] = word index of token i (-1 for delimiters)
        for wi, word in enumerate(words):
            ids = self._word_tokens(word)
            if not ids:
                continue
            if tokens and self.delim is not None:
                tokens.append(self.delim)
                owner.append(-1)
            tokens += ids
            owner += [wi] * len(ids)
        if not tokens:
            return None
        path = ctc_viterbi(log_probs, tokens, self.blank)
        if path is None:
            return None

        spans: list[list[float] | None] = [None] * len(words)
        owner_arr = np.asarray(owner)
        for t, tok in enumerate(path):
            if tok < 0 or owner_arr[tok] < 0:
                continue
            wi = owner_arr[tok]
            start, end = t * self.stride, (t + 1) * self.stride
            if spans[wi] is None:
                spans[wi] = [start, end]
            else:
                spans[wi][1] = end
        # Fill words without characters (e.g. a lone dash) from their neighbours.
        last_end = 0.0
        for wi in range(len(words)):
            if spans[wi] is None:
                spans[wi] = [last_end, last_end]
            last_end = spans[wi][1]
        return [(s, e) for s, e in spans]
