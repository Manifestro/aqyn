# Experiment Protocol

## Data

| Stage | Dataset | Hours | Purpose |
|---|---|---|---|
| Feasibility | LJSpeech | ~24 | single speaker, fast iteration (RTX 3060) |
| Main | LibriTTS-R | ~585 | multi-speaker, main comparisons |
| Later | other languages (TBD) | — | generalization, target languages |

Preprocessing:

1. Resample audio to 24 kHz (LJSpeech is 22.05 kHz; LibriTTS-R is already 24 kHz).
2. Encode with frozen Mimi and store 8 codebooks as `uint16`, shape `[T, 8]`.
3. Normalize text and convert it to phonemes (G2P). Characters are kept for the ablation.
4. For the text-in-stream variant only: word-level forced alignments (for example, Montreal Forced Aligner).

Held-out evaluation: the standard LJSpeech test split and LibriTTS-R `test-clean`, plus a fixed set of long texts (paragraphs) for the long-form tests.

## Phases

### Phase 0: pipeline and reference

- Tokenization with sanity checks (decode tokens back to audio and listen).
- **Mimi ceiling:** score plain Mimi encode/decode of the test set with every metric below.
- **Reference model:** a causal Transformer backbone at the same budget, with the same text encoder, cross-attention and depth module.

### Phase 1: main model

CfC + cross-attention + global conditioning. LJSpeech first, then LibriTTS-R.

### Phase 2: hybrid

- Add causal local self-attention: window of 50 or 100 frames (4 or 8 s).
- Placement: every 2nd block, every 3rd block, or top blocks only.

### Phase 3: ablations and controls

| Ablation | Values |
|---|---|
| Backbone mixer | CfC, LSTM, Mamba2, causal Transformer |
| Pure CfC (no cross-attention, text in stream) | text lead 0.5 / 1 / 2 s |
| Depth module | Transformer, CfC/GRU, delay pattern |
| Codebooks generated | 4, 8 |
| Text units | phonemes, characters |
| Alignment aids | none, guided attention loss, monotonic inference |

### Memory-horizon study

The goal is to measure where CfC's context runs out, and whether the limit is in **seconds** or in **recurrent steps**.

1. **Length buckets:** report every metric separately for utterances of 0–4, 4–8, 8–16 and 16+ seconds, and for paragraph-length long-form synthesis. Compare against the LSTM, Mamba2 and Transformer backbones.
2. **Steps vs seconds:** repeat each Mimi frame 1×, 2×, 4× and 8× (effective step rates of 12.5, 25, 50 and 100 Hz), so the same audio spans more recurrent steps. If quality drops at a fixed number of *steps*, the limit is in steps. If it drops at a fixed number of *seconds*, it is not.
3. **Probe task:** a synthetic recall task on Mimi token sequences (reproduce information seen `k` steps earlier), to measure the raw memory of each backbone without the TTS objective.
4. **Speaker drift:** speaker similarity of the first vs the last 4 s of long-form outputs.

## Metrics

| Metric | Description |
|---|---|
| Intelligibility | WER / CER of a strong ASR model (for example, Whisper large) on the synthesized audio |
| Naturalness | UTMOS (automatic); human MOS for the final models |
| Speaker similarity | cosine similarity of speaker embeddings, synthesized vs reference |
| Robustness | rate of skipped, repeated or truncated words on a hard-sentence set |
| Time to first audio | from text submission to the first 80 ms of audio |
| RTF | real-time factor of streaming synthesis on CPU (1 thread) and GPU |
| Memory | peak inference memory and per-stream state size |
| Train cost | GPU-hours to convergence |

## Rules for fair comparison

- Same data, text front end, text encoder, depth module, optimizer and step budget for every backbone.
- Backbone parameters matched within ±5%.
- Same sampling settings (temperature, top-k) for every model in a comparison.
- Report the mean of at least 2 seeds for headline numbers.
- All results come from streaming synthesis, unless explicitly labeled otherwise.

## Results template

| Run | Backbone | Params | WER | UTMOS | Spk-sim | TTFA | CPU RTF |
|---|---|---|---|---|---|---|---|
| mimi-ceiling | — | — | — | — | — | — | — |
| ref-transformer | causal Transformer | — | — | — | — | — | — |
| p1-cfc-xattn | CfC + cross-attn | — | — | — | — | — | — |
| p2-hybrid | CfC + cross-attn + local attn | — | — | — | — | — | — |

## Compute plan

| GPU | Tasks |
|---|---|
| RTX 3060 12 GB | unit tests, overfit-one-batch, LJSpeech runs at reduced size |
| T4 / L4 | Mimi tokenization, evaluation, latency benchmarks |
| A100 / H100 | LibriTTS-R training runs and controls |

CfC runs one sequential step per frame. To keep GPUs busy, use large batches, length bucketing, `torch.compile` / CUDA graphs, and a fused CfC kernel if profiling shows the loop is the bottleneck.
