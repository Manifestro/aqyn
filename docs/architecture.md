# Architecture

This document describes the planned model. Every number here is a starting point to be tested, not a final choice.

## 0. Guiding constraint

Earlier experiments suggest a CfC network holds context for only about **8 seconds**. The architecture therefore never asks CfC to remember anything long-range:

- **The text** is read through cross-attention, not stored in the recurrent state.
- **Speaker and style** are injected at every step, not memorized.
- **Long-range structure** beyond the CfC horizon comes from attention (Phase 2) or phrase-level synthesis.

## 1. Output: Mimi tokens

| Property | Value |
|---|---|
| Sample rate | 24 kHz |
| Frame rate | 12.5 Hz (80 ms per frame) |
| Codebooks generated | 8 (residual VQ) |
| Codebook size | 2048 |
| Decoder | Mimi, frozen, streaming |

- Codebook 0 is semantically distilled and carries most of the linguistic content. Codebooks 1–7 add acoustic detail.
- **Ceiling:** the quality of plain Mimi encode/decode at 8 codebooks bounds what any model in this project can reach.
- **Ablation:** generate 4 vs 8 codebooks to trade quality for speed.

## 2. Text front end

- **Input units:** phonemes from a G2P front end (default) or characters (ablation).
- **Text encoder:** non-causal Transformer, 4–6 layers, `d = 384–512`, ~8–10M parameters.
- In the main setup the whole utterance text is known before synthesis starts. Streaming *text input* (for example, from an LLM) is studied in Phase 3 through the text-in-stream variant.

## 3. Temporal backbone

One step per Mimi frame. **Input** at step `t`: the sum of the 8 codebook embeddings of frame `t−1` (`8 × 2048 × 512`, ~8.4M params), plus the speaker/style embedding.

Each block uses pre-norm residual connections:

```
x = x + CfC(Norm(x))                   # local dynamics, fixed-size state
x = x + CrossAttn(Norm(x), text)       # what to say next
x = x + FFN(Norm(x))                   # 512 → 2048 → 512
```

Phase 2 replaces some CfC mixers with causal local self-attention (sliding window of 50–100 frames, which is 4–8 s, with a rolling KV cache).

### 3.1 CfC mixer

- CfC cell (closed-form LTC approximation): hidden size 512, a backbone MLP and gated time-constant heads.
- `Δt = 1` per frame. Variants: default, no-gate, minimal.
- Stability: LayerNorm before the cell, gradient clipping, learned initial state.

### 3.2 Cross-attention to text

- Multi-head attention (8 heads) from the backbone state to the text encoder outputs.
- Cross-attention TTS can skip or repeat words. Mitigations to evaluate: a guided (diagonal) attention loss, location-aware attention, and monotonic alignment constraints at inference.

### 3.3 Global conditioning

- Single speaker (LJSpeech): none.
- Multi-speaker (LibriTTS-R): an embedding from a pretrained speaker encoder, or one learned from a reference clip, injected at every block through FiLM or addition.

## 4. Depth module

Predicts the 8 codebooks of frame `t` from the backbone output `h_t`:

- **Default:** a small causal Transformer over the codebook axis (4 layers, `d = 256`), as in Moshi. Codebook `k` is predicted given `h_t` and codebooks `0..k−1`.
- **Ablations:** a small CfC or GRU over the codebook axis, or a MusicGen-style delay pattern with parallel heads (no depth module).
- **End of speech:** a stop token or stop head on the backbone output.

## 5. Parameter budget (approximate)

| Component | Params |
|---|---|
| Text encoder | ~8–10M |
| Audio codebook embeddings (8 × 2048 × 512) | ~8.4M |
| CfC mixer (per block) | ~1.6M |
| Cross-attention (per block) | ~1.0M |
| FFN (per block) | ~2.1M |
| Backbone, 12 blocks | ~56M |
| Depth module + output heads | ~6–8M |
| **Total** | **~80M** |

Baseline backbones (LSTM, Mamba2, Transformer) are sized to match the backbone budget within ±5%.

## 6. Streaming synthesis

Per 80 ms frame:

1. The backbone takes the previous frame's tokens and updates its CfC state (and the KV cache of any local attention).
2. Cross-attention reads the relevant part of the text.
3. The depth module samples 8 tokens.
4. The Mimi decoder turns the frame into 80 ms of audio and sends it out.

Time to first audio is roughly one frame plus the text encoder pass. Memory per stream is constant for CfC and bounded by the window for local attention.

## 7. Variant: text in stream (Phase 3)

The cross-attention can be replaced by feeding time-aligned text tokens directly into the frame stream, with the text leading the audio by a fixed delay (1–2 s), as in Kyutai's delayed-streams TTS. This allows streaming text input, but requires word-level alignments for training and forces CfC to hold the upcoming text in its state. That makes it a direct test of the memory horizon.

## 8. Open questions

- Is the ~8 s horizon measured in seconds or in recurrent steps? (See the memory-horizon study in [experiments.md](experiments.md).)
- Is cross-attention alignment stable with a recurrent query, or does it need monotonic constraints?
- Which of the codebooks really need the depth module, and which can be predicted in parallel?
- How well does Mimi (trained mostly on English) represent other target languages?
