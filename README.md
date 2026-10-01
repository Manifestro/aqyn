# Mimi-CfC ASR

**Can Closed-form Continuous-time (CfC / LTC) networks do streaming speech recognition on top of Mimi codec tokens?**

This is an open research project by [Manifestro](https://github.com/Manifestro). We are building and evaluating a small (50–100M parameter), fully streaming ASR model that reads discrete tokens from the [Mimi](https://huggingface.co/kyutai/mimi) neural audio codec and processes them with CfC recurrent layers, first on their own and then combined with local attention.

> **Status:** research planning. No code yet. See the [roadmap](#roadmap).

---

## Motivation

- **Mimi tokens are known to work for ASR.** Kyutai's streaming speech-to-text models read Mimi tokens with Transformers. Mimi is causal and produces 12.5 frames per second (80 ms per frame) with 8 residual codebooks. The first codebook is distilled from a self-supervised speech model and carries phonetic content.
- **Recurrent models are known to work for streaming ASR.** LSTM-based RNN-Transducers have powered on-device recognition at the ~100M scale for years.
- **CfC networks have not been tested here, as far as we know.** CfC is the closed-form approximation of Liquid Time-Constant (LTC) networks. It is a recurrent cell with input-dependent time constants, constant memory per step and natural streaming. It has done well on small time-series and control tasks, but we know of no published results for large-vocabulary ASR at this scale.

Mimi's low frame rate helps a recurrent model a lot. A 10 s utterance is only **125 recurrent steps**, so the main cost of CfC (a nonlinear recurrence that cannot be computed with a parallel scan) stays manageable.

## Research questions

1. **RQ1:** Can a pure CfC encoder over Mimi tokens reach usable WER in streaming mode?
2. **RQ2:** How much does adding causal local (sliding-window) attention help a CfC encoder?
3. **RQ3:** Does CfC beat other recurrent baselines (LSTM, Mamba2) with the same parameters, data and training budget?
4. **RQ4:** How do accuracy, latency and inference cost trade off against a causal Transformer?

## Approach

```
audio (24 kHz, streaming)
  └─ Mimi encoder (frozen) ── 12.5 Hz, 8 codebooks × 2048 entries
       └─ token embeddings (one table per codebook, summed) ── d = 512
            └─ streaming encoder (16 blocks)
                 Phase 1: [CfC + FFN] × 16
                 Phase 2: [CfC + FFN, CfC + FFN, LocalAttn + FFN] repeated
                 └─ RNN-T head (prediction net + joiner), BPE vocab ≈ 1024
                    + auxiliary CTC loss on the encoder
```

Key design decisions (details in [docs/architecture.md](docs/architecture.md)):

- **Frozen Mimi.** Tokens are computed once and stored. LibriSpeech 960 h is under 1 GB of `uint16` tokens.
- **Transducer (RNN-T) instead of plain CTC.** At 12.5 Hz, fast speech can produce more characters than there are frames. RNN-T can emit several subword tokens per frame and is naturally streaming.
- **Strictly causal.** We test optional lookahead of 0, 1 and 2 frames (+0/80/160 ms).
- **Parameter-matched comparisons.** Every encoder variant targets the same budget (~70M total).

## Experiment plan

| Phase | Goal | Encoder |
|---|---|---|
| 0 | Data pipeline and Mimi tokenization | — |
| 1 | **Pure CfC**: does it learn at all, and how well? | CfC × 16 |
| 2 | **Hybrid**: CfC + causal local attention | CfC / LocalAttn |
| 3 | **Controls**: what does CfC add over other models? | LSTM, Mamba2, causal Transformer |
| 4 | Scaling, other languages, on-device inference | best of 1–3 |

Metrics: WER on LibriSpeech `test-clean` / `test-other`, emission latency, real-time factor on CPU and GPU, and peak memory. The full protocol is in [docs/experiments.md](docs/experiments.md).

## Hardware

| GPU | Use |
|---|---|
| RTX 3060 (12 GB) | development, smoke tests, small runs on `train-clean-100` |
| T4 / L4 | Mimi tokenization, evaluation, CPU/GPU latency benchmarks |
| A100 / H100 | main training runs (960 h and beyond) |

## Planned repository layout

```
configs/        experiment configs (one file per run)
src/mimicfc/
  data/         Mimi tokenization, datasets, BPE
  models/       CfC, local attention, baselines, RNN-T head
  train/        training loop, losses, logging
  eval/         WER, streaming latency, RTF
scripts/        entry points (tokenize, train, eval, export)
docs/           architecture, experiment protocol, results
```

## Roadmap

- [ ] Phase 0: Mimi tokenization of LibriSpeech, BPE tokenizer, data loaders
- [ ] Phase 0: causal Transformer + RNN-T reference baseline
- [ ] Phase 1: CfC encoder, streaming inference, first WER numbers
- [ ] Phase 1: ablations (codebooks, lookahead, CfC variants, depth/width)
- [ ] Phase 2: CfC + local attention hybrid
- [ ] Phase 3: LSTM / Mamba2 / Transformer controls
- [ ] Phase 4: other languages, scaling, CPU / mobile export
- [ ] Technical report with all results

## Contributing

Ideas, critique and experiments are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## References

- Hasani et al., *Closed-form continuous-time neural networks*, Nature Machine Intelligence, 2022.
- Hasani et al., *Liquid Time-constant Networks*, AAAI 2021.
- Défossez et al., *Moshi: a speech-text foundation model for real-time dialogue* (Mimi codec), 2024.
- Graves, *Sequence Transduction with Recurrent Neural Networks* (RNN-T), 2012.
- De et al., *Griffin: Mixing Gated Linear Recurrences with Local Attention*, 2024.
- Panayotov et al., *LibriSpeech: an ASR corpus based on public domain audio books*, ICASSP 2015.

## License

Code is released under the [Apache License 2.0](LICENSE). Mimi model weights are distributed by Kyutai under their own license (CC-BY 4.0 at the time of writing). Check upstream terms before redistributing weights or derived tokens.
