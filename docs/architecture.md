# Architecture

This document describes the planned model. Every number here is a starting point to be tested, not a final choice.

## 1. Input: Mimi tokens

| Property | Value |
|---|---|
| Sample rate | 24 kHz |
| Frame rate | 12.5 Hz (80 ms per frame) |
| Codebooks used | 8 (residual VQ) |
| Codebook size | 2048 |
| Causality | fully causal, streaming |

- Codebook 0 is semantically distilled and carries most of the phonetic information. Codebooks 1–7 are acoustic.
- **Embedding:** one table of size `2048 × d` per codebook. The 8 embeddings are summed into a single vector per frame. With `d = 512` this is about 8.4M parameters.
- **Ablation:** use 1, 4 or 8 codebooks. As a diagnostic upper bound, also feed the continuous pre-quantization Mimi latents.
- Mimi stays frozen for the whole project. Tokens are precomputed and stored as `uint16` arrays.

## 2. Encoder

All blocks use pre-norm residual connections:

```
x = x + Mixer(Norm(x))
x = x + FFN(Norm(x))
```

### 2.1 CfC mixer (Phase 1)

- CfC cell (closed-form LTC approximation): hidden size 512, a backbone MLP and gated time-constant heads.
- Time step `Δt = 1` per frame, since Mimi frames are uniform. Continuous-time behavior can still be useful for dropped or skipped frames.
- Variants to compare: default CfC, CfC without gate, minimal CfC.
- Stability: LayerNorm before the cell, gradient clipping, learned initial state or zero state.

### 2.2 Local attention mixer (Phase 2)

- Causal multi-head attention with a sliding window of 32–64 frames (2.5–5 s), 8 heads, and RoPE or ALiBi positions.
- Inference uses a fixed-size rolling KV cache, so memory stays bounded and streaming is preserved.
- Layout: `[CfC, CfC, Attn]` repeated, which replaces every third CfC mixer.

### 2.3 FFN

`512 → 2048 → 512` with GELU or SwiGLU.

### 2.4 Lookahead

The default is strictly causal (0 frames). An optional right context of 1–2 frames (80–160 ms) comes from a small causal convolution or a delayed output.

## 3. Output head

- **RNN-T** (transducer) over a BPE vocabulary of ~1024 tokens.
  - Prediction network: embedding plus a 1-layer LSTM (or a stateless prediction net).
  - Joiner: `tanh(W_enc·h_t + W_pred·g_u) → vocab + blank`.
- **Auxiliary CTC loss** on the encoder output (weight ~0.3) for faster, more stable convergence.
- Why not CTC alone: 12.5 frames/s is close to, or below, the character rate of fast speech. With BPE, CTC is viable and will be reported as an ablation.

## 4. Parameter budget (approximate, `d = 512`)

| Component | Params |
|---|---|
| Codebook embeddings (8 × 2048 × 512) | ~8.4M |
| CfC mixer (per block) | ~1.6M |
| Local attention mixer (per block) | ~1.0M |
| FFN (per block) | ~2.1M |
| Encoder, 16 CfC blocks | ~60M |
| RNN-T prediction net + joiner | ~3–4M |
| **Total (Phase 1)** | **~72M** |

Baseline encoders (LSTM, Mamba2, Transformer) are sized to match the encoder budget within ±5%.

## 5. Streaming inference

Per 80 ms frame:

1. Mimi encodes the new audio frame into 8 tokens.
2. The tokens are embedded and summed.
3. Each CfC layer updates its fixed-size state. Each attention layer appends to its rolling KV cache.
4. The RNN-T greedy (or beam) decoder emits zero or more tokens.

Algorithmic latency is 80 ms (one Mimi frame) plus any lookahead. State size is constant for CfC and bounded by the window for local attention.

## 6. Open questions

- Is the nonlinear CfC recurrence trainable at 16 layers deep without special initialization?
- Do the acoustic codebooks help ASR, or does codebook 0 alone suffice?
- How well does Mimi (trained mostly on English) represent other target languages?
- Is the training throughput of a sequential CfC loop acceptable at 12.5 Hz, or does it need a fused kernel?
