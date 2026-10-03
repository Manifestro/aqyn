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
3. Normalize text and map it to characters. Phonemes (G2P) are a planned ablation.
4. For the text-in-stream variant only: word-level forced alignments (for example, Montreal Forced Aligner).

Held-out evaluation: a fixed random LJSpeech split (seed 1234: 500 test, 100 val) and LibriTTS-R `test-clean`, plus a fixed set of long texts (paragraphs) for the long-form tests.

## Stage 1 experiments

1. **Mimi ceiling:** plain Mimi encode/decode of the test set, scored with every metric below.
2. **Transformer reference** first, to debug the whole pipeline before comparing architectures.
3. **Backbone comparison** at matched size and data: CfC, Mamba2, LSTM, causal Transformer, CfC + local attention.
4. **Pointer robustness:** rate of skipped / repeated words; effect of scheduled sampling and inference guards.
5. **Memory horizon:** metrics by utterance length (0–4, 4–8, 8–16, 16+ s) and 5-minute generations; frame repetition (1×, 2×, 4×, 8×) to tell whether the limit is in seconds or in steps.
6. **Speed control (optional):** tempo augmentation with CfC time step `dt = 1 / speed` vs scaling the pointer only.

**Decision rule, fixed in advance:** if CfC is within 10% of the best backbone on WER and speaker similarity while clearly better on memory and latency, keep CfC; otherwise use Mamba2.

## Metrics

| Metric | Description |
|---|---|
| Intelligibility | WER / CER of a strong ASR model (for example, Whisper large) on the synthesized audio |
| Naturalness | UTMOS (automatic); human MOS for the final models |
| Speaker similarity | cosine similarity of WavLM / ECAPA speaker embeddings, synthesized vs reference |
| Stress | accuracy on a homograph set, judged by listeners (ASR barely hears stress) |
| Long-form drift | voice and tempo stability over 5-minute generations |
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
