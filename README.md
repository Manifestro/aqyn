# Mimi-CfC TTS

**Can Closed-form Continuous-time (CfC / LTC) networks generate streaming speech as Mimi codec tokens?**

This is an open research project by [Manifestro](https://github.com/Manifestro). We are building and evaluating a small (50–100M parameter) streaming text-to-speech model. A CfC recurrent backbone generates [Mimi](https://huggingface.co/kyutai/mimi) codec tokens frame by frame, and attention supplies the long-range context that CfC cannot hold on its own.

> **Status:** Phase 0/1 code is ready (data pipeline, models, training, evaluation). No trained results yet. See the [roadmap](#roadmap).

---

## Motivation

- **Mimi is a natural target for streaming TTS.** It is causal, runs at 12.5 frames per second (80 ms per frame), and uses 8 residual codebooks. Each generated frame can be decoded into audio immediately.
- **Recurrent generation is cheap to stream.** A CfC cell keeps a fixed-size state: constant memory and compute per frame, no growing KV cache. That suits on-device and low-latency synthesis.
- **CfC has a limited memory horizon.** Our earlier experiments suggest CfC holds context for only about **8 seconds**. So we do **not** rely on CfC for long-range context. CfC models local dynamics such as articulation, transitions and short-range prosody. Attention and explicit conditioning carry everything longer: the text, the speaker and the style.
- **As far as we know, this is untested.** We know of no published work on CfC/LTC networks for codec-token speech generation.

An open question is whether the ~8 s limit is a limit in **seconds or in recurrent steps**. If it is in steps, Mimi's low frame rate stretches the same number of steps over much more audio. Measuring this is part of the project.

## Design principle: split memory by horizon

| What must be remembered | Horizon | Handled by |
|---|---|---|
| Articulation, phone transitions, rhythm | < 1 s | **CfC** |
| Intra-phrase prosody | 1–8 s | **CfC** + local self-attention |
| Text still to be spoken | whole utterance | **Cross-attention to text** (stored outside the state) |
| Speaker identity, style | entire output | **Global conditioning** injected at every step |
| Long-form (paragraphs) | > 8 s | phrase-level synthesis with carried context |

## Research questions

1. **RQ1:** Can a CfC backbone with cross-attention to text produce intelligible, natural speech as Mimi tokens in streaming mode?
2. **RQ2:** Where exactly does CfC's memory run out in TTS, and is the limit in seconds or in recurrent steps?
3. **RQ3:** How much do local self-attention and global conditioning compensate for the limited memory?
4. **RQ4:** Does CfC beat other recurrent backbones (LSTM, Mamba2) with the same parameters, data and training budget, and how does it compare with a causal Transformer in quality, latency and cost?

## Approach

```
text ── text encoder (non-causal) ──────────────┐
                                                ▼ cross-attention
prev. Mimi frame ─ 8 codebook embeddings ─ temporal backbone (CfC blocks) ─ depth module ─ 8 tokens ─ Mimi decoder ─ audio
                                                ▲                            (codebook by codebook)
speaker / style embedding ──────────────────────┘ (every step)
```

- **Temporal backbone:** CfC blocks advance one Mimi frame (80 ms) per step. Each block has a CfC mixer, cross-attention to the text and an FFN. Phase 2 adds causal local self-attention.
- **Depth module:** a small network that predicts the 8 codebooks of a frame one after another, conditioned on the backbone output.
- **Frozen Mimi:** audio is tokenized once. Training uses teacher forcing on the stored tokens.

Details are in [docs/architecture.md](docs/architecture.md).

## Experiment plan

| Phase | Goal |
|---|---|
| 0 | Data pipeline, Mimi tokenization, the Mimi resynthesis ceiling, a causal Transformer reference |
| 1 | **Main model:** CfC + cross-attention to text + global conditioning |
| 2 | **Hybrid:** add causal local self-attention, tune window size and placement |
| 3 | **Ablations and controls:** pure CfC, text-in-stream, LSTM / Mamba2 / Transformer backbones, memory-horizon study |
| 4 | Multi-speaker scaling, other languages, on-device inference |

Metrics: intelligibility (ASR-based WER/CER), naturalness (UTMOS, later human MOS), speaker similarity, time to first audio and real-time factor. The full protocol is in [docs/experiments.md](docs/experiments.md).

## Data

| Stage | Dataset | Notes |
|---|---|---|
| Feasibility | LJSpeech (~24 h, single speaker) | resampled to 24 kHz |
| Main | LibriTTS-R (~585 h, multi-speaker) | already 24 kHz |
| Later | other languages (TBD) | Mimi coverage to be checked first |

## Hardware

| GPU | Use |
|---|---|
| RTX 3060 (12 GB) | development, smoke tests, LJSpeech runs at reduced size |
| T4 / L4 | Mimi tokenization, evaluation, latency benchmarks |
| A100 / H100 | main training runs and controls |

## Quick start

Requires Python 3.10+ and a CUDA GPU (an RTX 3060 with 12 GB is enough for LJSpeech).

```bash
# 1. Install PyTorch for your CUDA version first (see pytorch.org), then:
pip install -e ".[eval,dev]"
pytest -q                                  # unit tests, run on CPU in seconds

# 2. Download LJSpeech (~2.6 GB) and encode it with Mimi (~10-20 min on a GPU)
python scripts/prepare_ljspeech.py --out data/ljspeech_tokens

# 3. Smoke test: a tiny model for 200 steps (a few minutes)
python scripts/train.py --config configs/debug.yaml

# 4. Main Phase 1 run (CfC + cross-attention)
python scripts/train.py --config configs/ljspeech_cfc.yaml
#    out of memory? lower the batch:   ... data.max_frames_per_batch=2000
#    resume after a stop:               ... --resume runs/ljspeech_cfc/last.pt

# 5. Listen and measure
python scripts/synthesize.py --ckpt runs/ljspeech_cfc/best.pt --text "Hello, this is a test." --out hello.wav
python scripts/evaluate.py --ceiling --data data/ljspeech_tokens --out results/mimi_ceiling
python scripts/evaluate.py --ckpt runs/ljspeech_cfc/best.pt --out results/ljspeech_cfc --utmos
```

Training logs go to `runs/<name>/log.jsonl`, checkpoints to `best.pt` / `last.pt`, and audio samples to `runs/<name>/samples/`.

### Configs

| Config | Backbone | Params | Phase |
|---|---|---|---|
| `configs/ljspeech_transformer.yaml` | causal Transformer, 14 blocks | 85.7M | 0 (reference) |
| `configs/ljspeech_cfc.yaml` | CfC, 12 blocks | 83.6M | 1 (main) |
| `configs/ljspeech_hybrid.yaml` | CfC + local attention (8 s window), 12 blocks | 81.5M | 2 |
| `configs/ljspeech_lstm.yaml` | LSTM, 11 blocks | 84.7M | 3 (control) |
| `configs/debug.yaml` | tiny CfC + local attention | 2.2M | smoke test |

All backbones are matched within ±5% of the CfC backbone. Any config value can be overridden from the command line, for example `train.lr=1e-4`.

## Repository layout

```
configs/              experiment configs (one file per run)
scripts/
  prepare_ljspeech.py download LJSpeech, resample, encode with Mimi
  train.py            training
  synthesize.py       text -> wav, with first-frame latency and RTF
  evaluate.py         ASR WER/CER, UTMOS, latency; also the Mimi ceiling
src/mimicfc/
  codec.py            frozen Mimi wrapper
  text.py             text normalization, character vocabulary
  data.py             token datasets, length-bucketed batching
  models/cfc.py       CfC cell (default / no_gate / pure)
  models/blocks.py    CfC, LSTM, attention mixers; cross-attention; streaming step()
  models/tts.py       text encoder, backbone, depth module, losses, generation
  train.py            training loop
tests/                streaming-vs-training parity, losses, generation
docs/                 architecture, experiment protocol
```

## Roadmap

- [x] Phase 0: Mimi tokenization of LJSpeech, text front end, data loaders
- [ ] Phase 0: LibriTTS-R preparation
- [ ] Phase 0: Mimi resynthesis ceiling and causal Transformer reference
- [x] Phase 1: CfC + cross-attention model and streaming synthesis (code)
- [ ] Phase 1: first trained models and samples
- [ ] Phase 2: local self-attention hybrid
- [ ] Phase 3: memory-horizon study, backbone controls, ablations
- [ ] Phase 4: multi-speaker scaling, other languages, CPU / mobile export
- [ ] Technical report with all results and audio samples

## Contributing

Ideas, critique and experiments are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## References

- Hasani et al., *Closed-form continuous-time neural networks*, Nature Machine Intelligence, 2022.
- Hasani et al., *Liquid Time-constant Networks*, AAAI 2021.
- Défossez et al., *Moshi: a speech-text foundation model for real-time dialogue* (Mimi codec, depth transformer), 2024.
- Copet et al., *Simple and Controllable Music Generation* (MusicGen, codebook delay patterns), 2023.
- De et al., *Griffin: Mixing Gated Linear Recurrences with Local Attention*, 2024.
- Koizumi et al., *LibriTTS-R: A Restored Multi-Speaker Text-to-Speech Corpus*, 2023.

## License

Code is released under the [Apache License 2.0](LICENSE). Mimi model weights are distributed by Kyutai under their own license (CC-BY 4.0 at the time of writing). Check upstream terms before redistributing weights, derived tokens or generated audio.
