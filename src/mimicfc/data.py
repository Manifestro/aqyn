"""Datasets over precomputed Mimi tokens.

Layout written by ``scripts/prepare_ljspeech.py``::

    root/
      vocab.json              character vocabulary
      {split}.jsonl           one line per utterance: id, text, offset, frames, speaker
      {split}_codes.npy       uint16 [total_frames, num_codebooks], utterances concatenated
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .text import CharVocab


def read_manifest(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class TokenDataset(Dataset):
    def __init__(self, root: str | Path, split: str, max_frames: int | None = None):
        self.root = Path(root)
        self.vocab = CharVocab.load(self.root / "vocab.json")
        items = read_manifest(self.root / f"{split}.jsonl")
        if max_frames:
            items = [it for it in items if it["frames"] <= max_frames]
        self.items = items
        self.codes_path = self.root / f"{split}_codes.npy"
        self._codes = None  # opened lazily so the dataset pickles cleanly into workers

    @property
    def codes(self) -> np.ndarray:
        if self._codes is None:
            self._codes = np.load(self.codes_path, mmap_mode="r")
        return self._codes

    def __len__(self) -> int:
        return len(self.items)

    def frames(self, i: int) -> int:
        return self.items[i]["frames"]

    def __getitem__(self, i: int) -> dict:
        it = self.items[i]
        codes = np.asarray(self.codes[it["offset"] : it["offset"] + it["frames"]], dtype=np.int64)
        return {
            "id": it["id"],
            "text": torch.tensor(self.vocab.encode(it["text"]), dtype=torch.long),
            "codes": torch.from_numpy(codes),
            "speaker": int(it.get("speaker", 0)),
        }


class FrameBudgetSampler(Sampler[list[int]]):
    """Batches of similar-length utterances whose padded size stays under a frame budget."""

    def __init__(self, dataset: TokenDataset, max_frames: int, shuffle: bool = True, seed: int = 0):
        self.lengths = [dataset.frames(i) for i in range(len(dataset))]
        self.max_frames = max_frames
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self._batches = self._make_batches()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self._batches = self._make_batches()

    def _make_batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        idx = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(idx)
        # Sort within large pools so batches are length-homogeneous but still random.
        pool = 100 * max(1, self.max_frames // 200)
        batches = []
        for start in range(0, len(idx), pool):
            chunk = sorted(idx[start : start + pool], key=lambda i: self.lengths[i])
            batch, longest = [], 0
            for i in chunk:
                if batch and max(longest, self.lengths[i]) * (len(batch) + 1) > self.max_frames:
                    batches.append(batch)
                    batch, longest = [], 0
                batch.append(i)
                longest = max(longest, self.lengths[i])
            if batch:
                batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)


def make_collate(pad_id: int):
    def collate(items: list[dict]) -> dict:
        b = len(items)
        t_max = max(len(it["text"]) for it in items)
        f_max = max(len(it["codes"]) for it in items)
        k = items[0]["codes"].shape[1]
        text = torch.full((b, t_max), pad_id, dtype=torch.long)
        codes = torch.zeros((b, f_max, k), dtype=torch.long)
        text_lens = torch.zeros(b, dtype=torch.long)
        code_lens = torch.zeros(b, dtype=torch.long)
        for i, it in enumerate(items):
            text[i, : len(it["text"])] = it["text"]
            codes[i, : len(it["codes"])] = it["codes"]
            text_lens[i] = len(it["text"])
            code_lens[i] = len(it["codes"])
        return {
            "ids": [it["id"] for it in items],
            "text": text,
            "text_lens": text_lens,
            "codes": codes,
            "code_lens": code_lens,
            "speaker": torch.tensor([it["speaker"] for it in items], dtype=torch.long),
        }

    return collate
