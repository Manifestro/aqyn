# Mimi-CfC TTS

**Can Closed-form Continuous-time (CfC / LTC) networks generate streaming speech as Mimi codec tokens?**

This is an open research project by [Manifestro](https://github.com/Manifestro). We are building and evaluating a small (50–100M parameter) streaming text-to-speech model. A CfC recurrent backbone generates [Mimi](https://huggingface.co/kyutai/mimi) codec tokens frame by frame, and attention supplies the long-range context that CfC cannot hold on its own.

> **Status:** research planning. No code yet. See the [roadmap](#roadmap).

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

## Planned repository layout

```
configs/        experiment configs (one file per run)
src/mimicfc/
  data/         Mimi tokenization, text processing, datasets
  models/       CfC, attention blocks, depth module, baselines
  train/        training loop, losses, logging
  eval/         intelligibility, naturalness, speaker similarity, latency
scripts/        entry points (tokenize, train, synthesize, eval, export)
docs/           architecture, experiment protocol, results
```

## Roadmap

- [ ] Phase 0: Mimi tokenization of LJSpeech and LibriTTS-R, text front end, data loaders
- [ ] Phase 0: Mimi resynthesis ceiling and causal Transformer reference
- [ ] Phase 1: CfC + cross-attention model, streaming synthesis, first samples
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
