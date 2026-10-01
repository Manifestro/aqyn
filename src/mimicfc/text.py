"""Text front end: normalization and a character vocabulary."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]

_ABBREVIATIONS = {
    "mr.": "mister",
    "mrs.": "misess",
    "dr.": "doctor",
    "st.": "saint",
    "co.": "company",
    "jr.": "junior",
    "maj.": "major",
    "gen.": "general",
    "drs.": "doctors",
    "rev.": "reverend",
    "lt.": "lieutenant",
    "hon.": "honorable",
    "sgt.": "sergeant",
    "capt.": "captain",
    "esq.": "esquire",
    "ltd.": "limited",
    "col.": "colonel",
    "ft.": "fort",
}

_ALLOWED = re.compile(r"[^a-z' .,;:!?\-]")
_SPACES = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase, expand common abbreviations, and keep a small character set.

    LJSpeech's normalized transcription column already spells out numbers.
    """
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    for abbr, full in _ABBREVIATIONS.items():
        text = re.sub(rf"\b{re.escape(abbr)}", full, text)
    text = text.replace('"', "").replace("(", ",").replace(")", ",")
    text = _ALLOWED.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


class CharVocab:
    def __init__(self, symbols: list[str]):
        self.symbols = SPECIALS + [s for s in symbols if s not in SPECIALS]
        self.index = {s: i for i, s in enumerate(self.symbols)}

    @classmethod
    def build(cls, texts: list[str]) -> CharVocab:
        chars = sorted({c for t in texts for c in normalize_text(t)})
        return cls(chars)

    @classmethod
    def load(cls, path: str | Path) -> CharVocab:
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.symbols, f, ensure_ascii=False, indent=0)

    def __len__(self) -> int:
        return len(self.symbols)

    @property
    def pad_id(self) -> int:
        return self.index[PAD]

    def encode(self, text: str) -> list[int]:
        unk = self.index[UNK]
        ids = [self.index.get(c, unk) for c in normalize_text(text)]
        return [self.index[BOS]] + ids + [self.index[EOS]]
