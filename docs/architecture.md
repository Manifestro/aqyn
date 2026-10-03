# Architecture

Every number here is a starting point to be tested, not a final choice.

## 0. Two speeds

| | Spinal cord | Cortex |
|---|---|---|
| Rate | 12.5 Hz, every Mimi frame | asynchronous, event-driven |
| Model | small recurrent network (CfC; Mamba2, LSTM, Transformer for comparison) | pretrained LLM |
| Output | Mimi tokens + control decisions | text (later with style tags) written into the text queue |

The two meet in the **text queue**: the cortex appends text, the spinal cord reads it through a sliding window and a word pointer that it moves itself. Stage 1 builds the spinal cord as a TTS model with exactly this interface; Stage 3 lets the cortex write into the same queue.

Earlier experiments suggest CfC holds context for only about **8 seconds**, so nothing long-range is kept in its state: the text comes through the window and the voice through a per-frame speaker embedding.

## 1. Mimi tokens and the codebook delay

| Property | Value |
|---|---|
| Sample rate / frame rate | 24 kHz / 12.5 Hz (80 ms) |
| Codebooks | 8 × 2048 (codebook 0 is semantic, 1–7 acoustic) |

Generation runs over **rows**. Row `s` holds the semantic token of frame `s` and the acoustic tokens of frame `s − d` (`d = acoustic_delay = 1`), so acoustic detail is predicted after the model has committed to the next semantic token. An utterance of `T` frames has `T + d` rows; undefined entries use a "no token" index and are masked from the loss.

## 2. Text: queue, window and pointer

- **Text** is normalized characters with words separated by spaces and an end token. `+` before a vowel marks stress (RUAccent convention, for Russian).
- **Text encoder:** character embeddings + a few local 1-D convolutions. Only local context is used, so appending text while speaking does not change what was already encoded (beyond the last few characters).
- **Window:** at each row the backbone sees `text_window = 32` characters starting `text_left = 8` characters before the first character of the current word.
- **Window cross-attention:** keys and values are computed once for the whole text and gathered per row, plus a learned bias per head and window offset. Used in every `cross_attn_every = 4`-th block. Cost per row is constant.
- **Pointer:** the current word index. A control head on the backbone output predicts how many words to advance before the next row (0..`max_advance` = 3).
- **Training targets** come from word alignments: the pointer at row `s` is the last word that has started by frame `min(s, T−1)`; the advance target is the pointer difference to the next row.
- **Inference guards:** speech may only end while the pointer is on the last word; the pointer is forced forward after `max_frames_per_word` rows on one word.

## 3. Backbone

Input per row: sum of the 8 codebook embeddings of the previous row (+ speaker embedding, + sinusoidal position for attention-only backbones). Blocks use pre-norm residuals:

```
x = x + Mixer(Norm(x))                        # CfC / LSTM / causal attention / local attention
x = x + WindowCrossAttn(Norm(x), text window) # in every 4th block
x = x + FFN(Norm(x))
```

Every block has a parallel `forward` for training and a `step` for streaming, tested to give identical results.

### CfC cell

Closed-form approximation of LTC (Hasani et al., 2022), modes `default`, `no_gate`, `pure`. The input projection runs over the whole sequence at once; only the recurrent part is in the time loop. The step is a pure function compiled once with `torch.compile` for all layers (falls back to eager if compilation is unavailable).

## 4. Heads

- **Depth module:** small causal Transformer over the codebook axis (4 layers, `d = 256`), predicting codebook `k` from the backbone output and codebooks `0..k−1` of the same row.
- **Control head:** advance the pointer by 0..3 words.
- **Stop head:** end of speech on the last row.

Loss: codebook cross-entropy (per-codebook weights) + stop BCE + advance cross-entropy.

## 5. Size

English fast-iteration configs (`d = 512`, 12 blocks): ~69M parameters, backbones matched within ±5%. The target configuration for multi-speaker and Russian/Kazakh data is larger: backbone 150–250M (`d ≈ 1024`, 12–16 blocks) and depth module 30–50M.

## 6. Voice (next step)

A speaker embedding computed from a 3–10 s reference is injected at every row, so the voice does not depend on CfC memory. Running the reference as a prefix and caching the resulting state is an optional extra for a better start, not the only carrier of identity.

## 7. Open questions

- Is the ~8 s horizon a limit in seconds or in recurrent steps?
- How often does the learned pointer skip or stall, and do scheduled sampling and the guards fix it?
- How well does Mimi represent Russian and Kazakh as spoken in Kazakhstan?
- Training throughput of CfC vs Mamba2 at `d ≈ 1024`.
