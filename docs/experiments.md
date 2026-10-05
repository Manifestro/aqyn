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

Training for all runs: one seed, bf16, AdamW, lr 3e-4 with 1k warm-up and cosine decay, 26k Mimi frames per batch, 20k steps on one 48 GB GPU (the CfC run on an NVIDIA L40, the Transformer run of the full comparison on an A40; an earlier Transformer run on the L40 reached the same validation losses). Sampling at temperature 0.8 / top-k 50, ASR with Whisper large-v3-turbo. Generation RTF excludes Mimi decoding.

Data: LibriTTS-R `train-clean-100` + `train-clean-360`, 146,016 training utterances (229.4 h, 1151 speakers) after dropping clips shorter than 0.5 s or longer than 20 s; 294 val and 492 test utterances from the same speakers.

### Full comparison: CfC vs causal Transformer

Both evaluated from the step-20k checkpoint with `scripts/pod/evaluate.sh` on an A40.

**Test set** (492 utterances x 2 sampling seeds; WER interval is a 95% bootstrap over utterances):

| Run | Params | WER | CER | UTMOS | Spk-sim | Duration ratio | First frame | GPU RTF |
|---|---|---|---|---|---|---|---|---|
| mimi-ceiling | — | 5.40% (4.71–6.12) | 2.41% | 3.83 | — | 1.000 | — | — |
| CfC, 12 blocks | 69.9M | 8.58% (7.94–9.21) | 4.17% | 3.04 | 0.911 | 1.011 | 23 ms | 0.25 |
| causal Transformer, 14 blocks | 70.9M | 8.37% (7.79–9.01) | 4.18% | 3.02 | 0.914 | 1.015 | 24 ms | 0.27 |

Paired bootstrap, WER(CfC) - WER(Transformer): +0.21 points, 95% interval -0.36 … +0.80. The test set cannot tell the two apart on intelligibility, naturalness or speaker similarity. Both are 3 WER points and 0.8 UTMOS below the codec ceiling.

**Long form** (`aqyn longform`: k consecutive test sentences joined into one text and generated as one stream; 30 paragraphs per length, 16 at k=16, 5 at k=48). Training clips are at most 20 s.

| | 1 sent. (~5 s) | 2 (~12 s) | 4 (~23 s) | 8 (~45 s) | 16 (~89 s) | 48 (~269 s) |
|---|---|---|---|---|---|---|
| CfC, WER | 6.49% | 6.69% | 8.63% | 7.97% | 8.44% | 7.49% |
| Transformer, WER | 5.92% | 7.32% | 9.29% | 15.19% | 28.20% | 56.79% |
| CfC, CER | 2.9% | 2.9% | 3.7% | 3.7% | 3.6% | 3.5% |
| Transformer, CER | 2.7% | 3.4% | 4.4% | 9.3% | 22.1% | 50.3% |
| CfC, voice drift | 0.940 | 0.927 | 0.930 | 0.934 | 0.941 | 0.930 |
| Transformer, voice drift | 0.937 | 0.921 | 0.928 | 0.919 | 0.903 | 0.858 |
| CfC, duration ratio | 0.995 | 1.070 | 1.107 | 1.092 | 1.109 | 1.135 |
| Transformer, duration ratio | 1.011 | 1.086 | 1.088 | 1.120 | 1.129 | 1.172 |

Voice drift is the speaker similarity between the first and the last 4 s of the generated audio (higher is steadier). No generation of either model hit the frame limit.

- CfC stays at 7–8.6% WER from 5 seconds to 4.5 minutes, with a steady voice. It was never trained on anything longer than 20 s.
- The Transformer matches it up to the training length and then degrades: twice the WER at 45 s, more than half of the words wrong at 4.5 minutes, and the voice drifts.
- Both speak about 10% slower on long texts than the summed references (which include no pauses between sentences).
- The longest lengths have few paragraphs (16 and 5), so the exact numbers there are rough; the size of the gap is not.

**Streaming cost** (`aqyn latency`, fp32, random weights, pod CPU with one thread and the A40):

| | Stream | Backbone ms / frame | Last 50 frames | Depth ms / frame | RTF | State | Peak memory |
|---|---|---|---|---|---|---|---|
| CfC, CPU | 10 s | 18.5 | 19.1 | 18.6 | 0.46 | 24 KB | 822 MB |
| CfC, CPU | 60 s | 16.6 | 16.3 | 16.5 | 0.41 | 24 KB | 844 MB |
| CfC, CPU | 300 s | 16.3 | 15.6 | 16.2 | 0.41 | 24 KB | 912 MB |
| Transformer, CPU | 10 s | 18.8 | 19.0 | 18.8 | 0.47 | 6.8 MB | 825 MB |
| Transformer, CPU | 60 s | 21.7 | 25.3 | 16.7 | 0.48 | 41 MB | 895 MB |
| Transformer, CPU | 300 s | 39.2 | 58.4 | 16.8 | 0.70 | 205 MB | 1162 MB |
| CfC, GPU | 300 s | 4.9 | 4.9 | 11.5 | 0.20 | 24 KB | 351 MB |
| Transformer, GPU | 300 s | 6.4 | 9.5 | 11.5 | 0.22 | 205 MB | 591 MB |

On short streams the two cost the same, and half of every frame goes to the depth module they share. With length nothing changes for CfC, while the Transformer's key/value cache grows by about 0.7 MB per second of audio and its CPU step is three times slower by the fifth minute. On a GPU the difference in time is small; the difference in state is the same.

**Training cost:** 0.60 s / step for the Transformer against about 1.4 for CfC on the same L40, i.e. 3.3 h against 8.0 h for 20k steps.

**Against the decision rule.** CfC is within the interval of the Transformer on WER and speaker similarity, and it is clearly ahead on long form and on per-stream state, so by the rule fixed in advance it stays. What this does not settle:

- The Transformer's long-form failure is most likely the sinusoidal frame positions past the 250 frames seen in training; attention with a local window (the hybrid config) or training on longer clips may behave differently. Not tested.
- No LSTM control: a plain recurrent backbone behind the same text window may hold up just as well, in which case the result is about the design, not about CfC.
- One training seed per backbone.

### First pass (100 utterances)

The first evaluation of the same CfC run and of an earlier Transformer run on the L40 used only the first 100 test utterances and one sampling seed. It is kept for the `best.pt` comparison; the full comparison above supersedes its WER numbers.

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

On those 100 utterances the two backbones were within one WER point (CfC checkpoints at 12k / 16k / 20k steps scored 7.5% / 8.6% / 7.8%), which the full test set confirmed to be noise. The first 100 utterances are slightly easier than the rest: the ceiling is 4.9% there and 5.4% on all 492.

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

- LSTM control and the hybrid with the same recipe; a second training seed.
- A Transformer with relative or windowed positions, to separate "attention" from "absolute positions" in the long-form failure.
- The 0.8 UTMOS gap to the codec ceiling, common to both backbones.
- Regularize the stop / advance heads.

## Comparison protocol

The question: is CfC better than the alternatives at anything, or only not worse? One command per backbone (`scripts/pod/queue.sh`), the same recipe as above, with `best.pt` now selected by validation codes loss.

| What | How | Decides |
|---|---|---|
| Quality on the test set | all 492 test utterances x 2 sampling seeds: WER / CER, UTMOS, speaker similarity (WavLM x-vector cosine to the Mimi-reconstructed reference) | is any backbone better on quality; paired bootstrap against the Transformer |
| Long form | paragraphs of 1, 2, 4, 8, 16 and 48 joined test sentences (about 6 s to 5 min), one stream each: WER, duration ratio, runs that hit the frame limit, voice drift | does a backbone fall apart beyond the 20 s it was trained on |
| Streaming cost | `aqyn latency` on one CPU thread and on the GPU, streams of 10 s, 1 min, 5 min: ms per frame, RTF, state size, peak memory | the "clearly better on memory and latency" half of the decision rule |
| Training cost | s / step and GPU-hours from the logs | what the choice costs |
| Seeds | a second training seed for CfC and Transformer if the first pass is close | whether a gap survives a re-run |

Backbones: causal Transformer (reference), CfC, LSTM (control: is it CfC or just "any recurrent net with the text window"), CfC + local attention. Mamba2 is not implemented yet.

Reading the outcome, fixed in advance:

- CfC wins if quality is within the interval of the best backbone **and** it is clearly ahead on long form or on streaming cost.
- If LSTM matches CfC everywhere, the result is about the text window and the recurrent state, not about CfC; LSTM is then the simpler choice.
- If the Transformer is as good on long form and the cost gap does not matter at the target stream length, CfC only costs training time (2.3x per step) and is dropped.

## Compute plan

| GPU | Tasks |
|---|---|
| RTX 3060 12 GB | unit tests, overfit-one-batch, LJSpeech runs at reduced size |
| T4 / L4 | Mimi tokenization, evaluation, latency benchmarks |
| A100 / H100 | LibriTTS-R training runs and controls |

CfC runs one sequential step per frame. To keep GPUs busy, use large batches, length bucketing, `torch.compile` / CUDA graphs, and a fused CfC kernel if profiling shows the loop is the bottleneck.
