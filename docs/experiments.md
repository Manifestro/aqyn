# Experiment Protocol

## Data

| Stage | Dataset | Hours | Purpose |
|---|---|---|---|
| Dev | LibriSpeech `train-clean-100` | 100 | debugging, fast iteration (RTX 3060) |
| Main | LibriSpeech 960 h | 960 | main comparisons |
| Later | other datasets / languages (TBD) | — | generalization, target languages |

Evaluation sets: LibriSpeech `dev-clean`, `dev-other`, `test-clean`, `test-other`.

Preprocessing:

1. Resample audio to 24 kHz.
2. Encode it with frozen Mimi and store 8 codebooks as `uint16`, shape `[T, 8]`.
3. Normalize transcripts and train a BPE tokenizer (~1024 tokens) on the training transcripts only.

## Phases

### Phase 0: pipeline and reference

- Tokenization pipeline with sanity checks (decode tokens back to audio and listen).
- Reference model: causal Transformer encoder + RNN-T at the same parameter budget. This sets the target to beat.

### Phase 1: pure CfC

Main run: `CfC × 16` + RNN-T on 960 h.

Ablations (on 100 h first, then the best ones on 960 h):

| Ablation | Values |
|---|---|
| Codebooks | 1, 4, 8 |
| Lookahead | 0, 1, 2 frames |
| CfC variant | default, no-gate, minimal |
| Depth × width | 12×576, 16×512, 24×416 |
| Loss | RNN-T, RNN-T + CTC, CTC only |

### Phase 2: hybrid

- `[CfC, CfC, LocalAttn]` layout, window 32 / 64 frames.
- Attention placement: every 2nd, every 3rd, or only the top layers.

### Phase 3: controls

Swap the mixer and keep everything else the same:

- LSTM
- Mamba2
- causal Transformer (full context and local window)

## Metrics

| Metric | Description |
|---|---|
| WER | `test-clean`, `test-other`, greedy and beam decoding |
| Emission latency | delay between a word's end in the audio and the emission of its last token (from forced alignments) |
| RTF | real-time factor, streaming frame by frame, on CPU (1 thread) and GPU |
| Memory | peak inference memory and per-stream state size |
| Params | total and encoder-only |
| Train cost | GPU-hours to convergence |

## Rules for fair comparison

- Same data, tokenizer, head, optimizer and step budget for every encoder variant.
- Encoder parameters matched within ±5%.
- Report the mean of at least 2 seeds for headline numbers.
- All results are reported in streaming mode, never in full-utterance mode, unless explicitly labeled.

## Results template

| Run | Encoder | Params | Lookahead | test-clean WER | test-other WER | Latency | CPU RTF |
|---|---|---|---|---|---|---|---|
| ref-transformer | causal Transformer | — | 0 | — | — | — | — |
| p1-cfc | CfC × 16 | — | 0 | — | — | — | — |
| p2-hybrid | CfC + LocalAttn | — | 0 | — | — | — | — |

## Compute plan

| GPU | Tasks |
|---|---|
| RTX 3060 12 GB | unit tests, overfit-one-batch, 100 h runs at reduced width |
| T4 / L4 | Mimi tokenization, evaluation, latency benchmarks |
| A100 / H100 | 960 h training runs and controls |

CfC runs one sequential step per frame. To keep GPUs busy, use large batches, length bucketing, `torch.compile` / CUDA graphs, and a fused CfC kernel if profiling shows the loop is the bottleneck.
