# Experiment Protocol

## Data

| Stage | Dataset | Hours | Purpose |
|---|---|---|---|
| Feasibility | LJSpeech | ~24 | single speaker, smoke tests (RTX 3060) |
| Main | LibriTTS-R | ~585 (229 used so far: `train-clean-100` + `train-clean-360`) | multi-speaker, main comparisons |
| Later | other languages (TBD) | — | generalization, target languages |

Preprocessing:

1. Resample audio to 24 kHz (LJSpeech is 22.05 kHz; LibriTTS-R is already 24 kHz).
2. Encode with frozen Mimi and store 8 codebooks as `uint16`, shape `[T, 8]`.
3. Normalize text and map it to characters. Phonemes (G2P) are a planned ablation.
4. Word start times from a character CTC model (wav2vec2 for English) with a Viterbi pass over the transcript; they train the advance head.

Held-out evaluation: a fixed random split over utterances (seed 1234: 500 test, 100 val for LJSpeech, 300 val for LibriTTS-R). The model uses a speaker table, so every LibriTTS-R speaker is seen in training and `test-clean` (unseen speakers) is not used yet. A fixed set of long texts (paragraphs) for the long-form tests is still to be added.

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

## Results

All runs so far: one seed, one NVIDIA L40 (48 GB), bf16, AdamW, lr 3e-4 with 1k warm-up and cosine decay, 26k Mimi frames per batch, 20k steps. Evaluation: 100 test utterances, sampling at temperature 0.8 / top-k 50, ASR with Whisper large-v3-turbo. RTF excludes Mimi decoding.

### LibriTTS-R: CfC vs causal Transformer

Data: `train-clean-100` + `train-clean-360`, 146,016 training utterances (229.4 h, 1151 speakers) after dropping clips shorter than 0.5 s or longer than 20 s; 294 val and 492 test utterances from the same speakers.

| Run | Backbone | Params | WER | CER | Duration ratio | First frame | GPU RTF | s / step | Train time |
|---|---|---|---|---|---|---|---|---|---|
| mimi-ceiling | — | — | 4.89% | 2.18% | 1.000 | — | — | — | — |
| libritts_r_cfc, `last` | CfC, 12 blocks | 69.9M | 7.76% | 3.42% | 1.005 | 28.9 ms | 0.268 | ~1.4 | 8.0 h |
| libritts_r_transformer, `last` | causal Transformer, 14 blocks | 70.9M | 8.79% | 4.15% | 1.018 | 28.9 ms | 0.273 | 0.60 | 3.3 h |
| libritts_r_cfc, `best` (step 8k) | CfC | 69.9M | 11.24% | 5.28% | 0.971 | 28.6 ms | 0.266 | | |
| libritts_r_transformer, `best` (step 8k) | causal Transformer | 70.9M | 12.92% | 6.58% | 0.994 | 28.9 ms | 0.275 | | |

Validation at step 20k (teacher forcing):

| Run | Train codes | Val codes | Val cb0 | Val stop | Val advance loss | Val advance acc. |
|---|---|---|---|---|---|---|
| libritts_r_cfc | 4.27 | 4.21 | 1.59 | 0.33 | 0.59 | 0.89 |
| libritts_r_transformer | 4.21 | 4.17 | 1.58 | 0.23 | 0.68 | 0.89 |

What this does and does not show:

- **Parity, not a win.** The two backbones are within one WER point on 100 utterances and one seed, which is inside the noise (CfC checkpoints at 12k / 16k / 20k steps scored 7.5% / 8.6% / 7.8%). The Transformer is slightly ahead on validation codes loss (by 0.03-0.04 throughout training), CfC slightly ahead on WER.
- **Training cost.** The Transformer trains 2.3x faster per step; CfC runs one sequential step per frame.
- **Inference.** Streaming generation costs the same for both on a GPU. CPU RTF, memory and per-stream state size, where CfC is expected to differ, are not measured yet.
- **Not measured:** UTMOS (the eval skipped it: `torchaudio` was missing), speaker similarity, long-form drift, pointer robustness.

### The stop and advance heads overfit

In both runs the codes loss on validation keeps falling to the last step and stays at or below the training value, but the stop and advance losses on validation rise from about step 5k (advance: 0.22 -> 0.59 for CfC, while 0.03 on train). Advance accuracy on validation stays at 0.89 the whole time, and the duration of generated speech matches the reference, so the heads become overconfident rather than more wrong.

Consequence: `best.pt`, chosen by the total validation loss in these runs, froze at step 8k in both and is clearly worse than `last.pt` (11.2% vs 7.8% WER for CfC). Training now selects `best.pt` by validation codes loss instead; in both runs above that would have been the last step.

### LJSpeech: too small for this model

The 69M CfC model overfits LJSpeech within a few thousand steps at this batch size (43 batches per epoch):

| Step | Train loss | Val loss | Val codes | Val cb0 |
|---|---|---|---|---|
| 2000 | 4.89 | 6.08 | 5.02 | 2.14 |
| 3000 | 4.37 | 6.01 | 4.90 | 2.22 |
| 4000 | 4.03 | 6.17 | 5.05 | 2.58 |

The run was stopped at step 4k and not evaluated. LJSpeech stays useful for smoke tests only.

### Open items from these runs

- Second seed and a larger test set before any claim about CfC vs Transformer.
- LSTM and hybrid (CfC + local attention) controls with the same recipe.
- Regularize the stop / advance heads.
- UTMOS, speaker similarity, CPU RTF and memory.

## Compute plan

| GPU | Tasks |
|---|---|
| RTX 3060 12 GB | unit tests, overfit-one-batch, LJSpeech runs at reduced size |
| T4 / L4 | Mimi tokenization, evaluation, latency benchmarks |
| A100 / H100 | LibriTTS-R training runs and controls |

CfC runs one sequential step per frame. To keep GPUs busy, use large batches, length bucketing, `torch.compile` / CUDA graphs, and a fused CfC kernel if profiling shows the loop is the bottleneck.
