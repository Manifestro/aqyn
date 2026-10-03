# Aqyn

*Aqyn (акын) is a Kazakh improvising poet-singer who composes and performs in real time.*

**A two-speed conversational speech model: a fast recurrent "spinal cord" that listens and speaks every 80 ms, and a slow, swappable "cortex" (an LLM) that decides what to say.**

This is an open research project by [Manifestro](https://github.com/Manifestro). The fast part runs on [Mimi](https://huggingface.co/kyutai/mimi) codec tokens at 12.5 Hz and is built around Closed-form Continuous-time (CfC / LTC) networks, compared against Mamba2, LSTM and Transformer backbones.

> **Status:** Stage 1 (streaming TTS on the spinal cord) is implemented and runs end to end on small data. No trained results yet. English first for fast iteration; Russian and Kazakh data are being collected.

---

## Idea

Full-duplex speech models today are one large Transformer that listens, thinks and speaks every 80 ms, so intelligence is paid for in latency. Aqyn splits the model in two:

| | Spinal cord (fast) | Cortex (slow) |
|---|---|---|
| Rate | every Mimi frame, 12.5 Hz | asynchronous, event-driven |
| Model | small recurrent network (CfC; Mamba2 as fallback) | any pretrained LLM, optionally with LoRA |
| Does | speaking, prosody, backchannels, pauses, interruptions, turn-taking | reasoning, content, tool calls |
| Interface | reads a **text queue** through a sliding window and a word pointer | writes text (later: with style tags) into the queue |

The cortex can be upgraded without retraining the spinal cord. While a tool call runs, the spinal cord keeps the conversation going ("one second, let me check…").

Why CfC here: it is small, reactive, keeps constant memory per frame, and comes from control problems. Our earlier experiments suggest CfC holds context for only about **8 seconds**, so the design never asks it to remember anything long: the text comes through the window, the voice through a per-frame speaker embedding.

## Stage 1: streaming TTS on the spinal cord

```
text queue ─ char embeddings + local convs ─┐
                                           ▼  window cross-attention (W chars around the current word)
prev. row ─ 8 codebook embeddings ─ backbone (CfC / LSTM / attn / local) ─┬─ depth module ─ semantic token + 7 acoustic (1-frame delay) ─ Mimi decoder
                                                                         ├─ control head: advance the word pointer by 0..3
                                                                         └─ stop head
```

- **Text window instead of attention over the whole text.** At every frame the backbone sees a fixed window of characters around the current word, so a step costs O(1). A control head decides when to move to the next word; it is trained from word alignments. This queue is the interface the cortex will write into in Stage 3.
- **Stress marks.** Text may contain `+` before a stressed vowel (RUAccent convention) for Russian; the model learns to follow it.
- **Acoustic delay.** Acoustic codebooks lag the semantic one by one frame, as in Moshi.
- **Word alignment** comes from a character CTC model (wav2vec2 for English; MMS covers Russian and Kazakh) with a Viterbi pass over the transcript.
- **Generation guards.** Speech can only end on the last word, and the pointer is forced forward after too long on one word.

Details are in [docs/architecture.md](docs/architecture.md).

## Plan

| Stage | Goal | Useful on its own as |
|---|---|---|
| 1 | Streaming TTS on the spinal cord; CfC vs Mamba2 vs LSTM vs Transformer | a streaming TTS |
| 2 | Hearing and turn-taking: backchannels, interruptions, yielding the floor (including synthetic dialogues) | a reactive voice front end |
| 3 | Connect the cortex through the text queue; tool calls during conversation | the full system |

Stage 1 metrics: WER through ASR, speaker similarity, stress accuracy on homographs (human listening), 5-minute generations without voice or tempo drift, time to first audio, RTF. Decision rule fixed in advance: if CfC is within 10% of the best backbone on WER and speaker similarity while clearly better on memory and latency, keep it; otherwise use Mamba2. See [docs/experiments.md](docs/experiments.md).

## Data

| Stage | Dataset | Notes |
|---|---|---|
| Fast iteration (now) | LJSpeech (~24 h, English, single speaker) | resampled to 24 kHz |
| Multi-speaker English | LibriTTS-R (~585 h) | already 24 kHz |
| Target | Russian and Kazakh as spoken in Kazakhstan | being collected; Mimi coverage checked first |

Dataset licenses are checked one by one before use; non-commercial sets are kept out of anything shipped.

## Hardware

| GPU | Use |
|---|---|
| RTX 3060 (12 GB) | development, smoke tests, LJSpeech runs at reduced size |
| T4 / L4 | Mimi tokenization, evaluation, latency benchmarks |
| A100 / H100 | main training runs and controls |

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU (an RTX 3060 with 12 GB is enough for LJSpeech). On Linux and Windows, uv installs the CUDA 12.8 build of PyTorch automatically.

```bash
uv sync                                     # create .venv and install everything
uv run pytest -q                            # unit tests (CPU, seconds)

uv run aqyn prepare                         # download LJSpeech (~2.6 GB), encode with Mimi
uv run aqyn bench --config configs/ljspeech_cfc.yaml   # speed and memory per batch size on your GPU
uv run aqyn train --config configs/debug.yaml          # smoke test, a few minutes
uv run aqyn train --config configs/ljspeech_cfc.yaml   # main Stage 1 run (CfC)

uv run aqyn synth --ckpt runs/ljspeech_cfc/best.pt --text "Hello, this is a test." --out hello.wav
uv run aqyn eval --ceiling --data data/ljspeech_tokens --out results/mimi_ceiling
uv run aqyn eval --ckpt runs/ljspeech_cfc/best.pt --out results/ljspeech_cfc --utmos
```

- Any config value can be overridden at the end of the command, e.g. `data.max_frames_per_batch=6000` (use `aqyn bench` to pick it).
- Resume a stopped run with `--resume runs/<name>/last.pt`.
- `uv run aqyn <command> --help` shows all options.

Training logs go to `runs/<name>/log.jsonl`, checkpoints to `best.pt` / `last.pt`, and audio samples to `runs/<name>/samples/`.

### Training speed

CfC is a nonlinear recurrence, so it runs one frame at a time (12 blocks × ~100 frames = ~1200 sequential steps per batch). To keep the GPU busy:

- **Use the largest batch that fits.** The number of sequential steps does not grow with batch size, so throughput scales almost linearly. `aqyn bench` measures this.
- **Compiled CfC step (on by default on CUDA).** `torch.compile` fuses the ~15 small kernels of each step into a few. If compilation is unavailable (for example, Windows without Triton), training falls back to the eager step automatically. Disable with `train.compile_cfc=false`.

### Configs

| Config | Backbone | Params | Role |
|---|---|---|---|
| `configs/ljspeech_transformer.yaml` | causal Transformer, 14 blocks | 70.3M | reference |
| `configs/ljspeech_cfc.yaml` | CfC, 12 blocks | 69.3M | main |
| `configs/ljspeech_hybrid.yaml` | CfC + local attention (8 s window), 12 blocks | 67.2M | hybrid |
| `configs/ljspeech_lstm.yaml` | LSTM, 11 blocks | 71.4M | control |
| `configs/debug.yaml` | tiny CfC + local attention | 2.2M | smoke test |

All backbones are matched within ±5% of the CfC backbone. Any config value can be overridden from the command line, for example `train.lr=1e-4`.

## Repository layout

```
configs/              experiment configs (one file per run)
src/aqyn/
  cli.py              `aqyn` command: prepare / train / synth / eval / bench
  prepare.py          download LJSpeech, resample, encode with Mimi, align words
  align.py            CTC forced alignment of words (wav2vec2 / MMS)
  train.py            training loop
  synthesize.py       text -> wav, with first-frame latency and RTF
  evaluate.py         ASR WER/CER, UTMOS, latency; also the Mimi ceiling
  bench.py            speed and memory benchmark on synthetic batches
  codec.py            frozen Mimi wrapper
  text.py             text normalization, character vocabulary
  data.py             token datasets, length-bucketed batching
  models/cfc.py       CfC cell (default / no_gate / pure), compiled step
  models/blocks.py    CfC, LSTM, attention mixers; text-window cross-attention; streaming step()
  models/tts.py       text encoder, word pointer, codebook delay, depth module, losses, generation
tests/                streaming-vs-training parity, pointer targets, delay, losses, generation
docs/                 architecture, experiment protocol
```

## Roadmap

- [x] Data: LJSpeech download, Mimi tokenization, CTC word alignment
- [x] Stage 1 model: text window + word pointer, acoustic delay, control and stop heads
- [x] Training, synthesis, evaluation (ASR WER/CER, UTMOS, latency, Mimi ceiling), speed benchmark
- [ ] Stage 1: first trained models and samples on LJSpeech (CfC vs Transformer vs LSTM)
- [ ] Stage 1: Mamba2 backbone; multi-speaker English (LibriTTS-R) with speaker conditioning
- [ ] Stage 1: Russian and Kazakh data, stress marks, 5-minute drift test
- [ ] Stage 2: hearing and turn-taking
- [ ] Stage 3: cortex through the text queue, tool calls
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
