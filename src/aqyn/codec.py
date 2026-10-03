"""Thin wrapper around the frozen Mimi codec from Hugging Face transformers."""

from __future__ import annotations

import numpy as np
import soundfile as sf
import soxr
import torch

MIMI_ID = "kyutai/mimi"
SAMPLE_RATE = 24_000
FRAME_RATE = 12.5
SAMPLES_PER_FRAME = 1920


def load_audio(path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load an audio file as mono float32 at ``target_sr``."""
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != target_sr:
        wav = soxr.resample(wav, sr, target_sr, quality="HQ")
    return wav.astype(np.float32)


def save_audio(path: str, wav: np.ndarray | torch.Tensor, sr: int = SAMPLE_RATE) -> None:
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().float().cpu().numpy()
    sf.write(path, np.clip(wav.reshape(-1), -1.0, 1.0), sr)


class Mimi:
    def __init__(self, device: str | torch.device = "cpu", num_codebooks: int = 8):
        from transformers import MimiModel

        self.device = torch.device(device)
        self.num_codebooks = num_codebooks
        self.model = MimiModel.from_pretrained(MIMI_ID).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, wav: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Mono 24 kHz waveform ``[samples]`` -> codes ``[frames, num_codebooks]`` (long, CPU)."""
        x = torch.as_tensor(wav, dtype=torch.float32, device=self.device).view(1, 1, -1)
        out = self.model.encode(x, num_quantizers=self.num_codebooks)
        return out.audio_codes[0].transpose(0, 1).long().cpu()

    @torch.inference_mode()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Codes ``[frames, num_codebooks]`` -> waveform ``[samples]`` (float, CPU)."""
        c = codes.to(self.device).long().transpose(0, 1).unsqueeze(0)
        out = self.model.decode(c)
        return out.audio_values[0, 0].float().cpu()
