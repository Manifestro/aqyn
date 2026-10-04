# Aqyn

*Aqyn (акын) is a Kazakh improvising poet-singer who composes and performs in real time.*

**A two-speed conversational speech model: a fast recurrent "spinal cord" that listens and speaks every 80 ms, and a slow, swappable "cortex" (an LLM) that decides what to say.**

This is an open research project by [Manifestro](https://github.com/Manifestro). The fast part runs on [Mimi](https://huggingface.co/kyutai/mimi) codec tokens at 12.5 Hz and is built around Closed-form Continuous-time (CfC / LTC) networks, compared against Mamba2, LSTM and Transformer backbones.

> **Status:** Stage 1 (streaming TTS on the spinal cord) is implemented and trained on LibriTTS-R. In the first matched comparison the CfC backbone is on par with a causal Transformer (WER 7.8% vs 8.8%, Mimi ceiling 4.9%; one seed, 100 utterances). English first for fast iteration; Russian and Kazakh data are being collected.

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

## First results

Both models: ~70M parameters, LibriTTS-R `train-clean-100` + `train-clean-360` (229 h, 1151 speakers), 20k steps, batches of 26k Mimi frames, one NVIDIA L40. Scores are on 100 held-out test utterances, transcribed with Whisper large-v3-turbo.

| Model | WER | CER | Val codes loss | First frame | RTF (GPU) | Train time |
|---|---|---|---|---|---|---|
| Mimi ceiling (encode/decode only) | 4.9% | 2.2% | — | — | — | — |
| CfC, 12 blocks | 7.8% | 3.4% | 4.21 | 29 ms | 0.27 | 8.0 h |
| Causal Transformer, 14 blocks | 8.8% | 4.1% | 4.17 | 29 ms | 0.27 | 3.3 h |

One seed and 100 utterances: the WER gap is within noise, so read this as "CfC is not worse", not "CfC is better". Naturalness (UTMOS) and speaker similarity are not measured yet. Full numbers and what went wrong on the way are in [docs/experiments.md](docs/experiments.md#results).

Stage 1 metrics: WER through ASR, speaker similarity, stress accuracy on homographs (human listening), 5-minute generations without voice or tempo drift, time to first audio, RTF. Decision rule fixed in advance: if CfC is within 10% of the best backbone on WER and speaker similarity while clearly better on memory and latency, keep it; otherwise use Mamba2. See [docs/experiments.md](docs/experiments.md).

## Data

| Stage | Dataset | Notes |
|---|---|---|
| Smoke tests | LJSpeech (~24 h, English, single speaker) | resampled to 24 kHz; a 70M model overfits it within ~3k steps |
| Multi-speaker English (now) | LibriTTS-R, `train-clean-100` + `train-clean-360` (~229 h after filtering, 1151 speakers) | already 24 kHz; `train-other-500` not used yet |
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
uv run aqyn train --config configs/debug.yaml          # smoke test, a few minutes
```

The main runs use LibriTTS-R (~37 GB to download, about an hour of tokenization on one GPU):

```bash
uv run aqyn prepare --dataset libritts_r --out data/libritts_r_tokens --num-val 300
uv run aqyn bench --config configs/libritts_r_cfc.yaml   # speed and memory per batch size on your GPU
uv run aqyn train --config configs/libritts_r_cfc.yaml data.max_frames_per_batch=26000 \
    train.max_steps=20000 train.warmup_steps=1000 train.eval_every=1000 train.save_every=500 train.sample_every=2000

uv run aqyn synth --ckpt runs/libritts_r_cfc/last.pt --speaker 0 --text "Hello, this is a test." --out hello.wav
uv run aqyn eval --ceiling --data data/libritts_r_tokens --out results/ceiling --num 100
uv run aqyn eval --ckpt runs/libritts_r_cfc/last.pt --out results/cfc_last --num 100
```

The second block is the exact recipe behind the numbers above (the Transformer run only swaps the config); 26k frames per batch needs a 48 GB GPU.

- Any config value can be overridden at the end of the command, e.g. `data.max_frames_per_batch=6000` (use `aqyn bench` to pick it).
- Stop a run with Ctrl-C (or `kill`): it finishes the current step and writes `last.pt`. Resume with `--resume runs/<name>/last.pt` and the same config/overrides; it continues from the same step and the same place in the epoch.
- `uv run aqyn <command> --help` shows all options.

- The speaker table is sized from the data, so the same config works for single- and multi-speaker corpora.

Training logs go to `runs/<name>/log.jsonl`, checkpoints to `best.pt` / `last.pt`, and audio samples to `runs/<name>/samples/`. `best.pt` is picked by the total validation loss, which the stop and advance heads dominate late in training; so far `last.pt` has been the better model (see [docs/experiments.md](docs/experiments.md#results)).

### Training speed

CfC is a nonlinear recurrence, so it runs one frame at a time (12 blocks × ~100 frames = ~1200 sequential steps per batch). To keep the GPU busy:

- **Use the largest batch that fits.** The number of sequential steps does not grow with batch size, so throughput scales almost linearly. `aqyn bench` measures this.
- **Compiled CfC step (on by default on CUDA).** `torch.compile` fuses the ~15 small kernels of each step into a few. If compilation is unavailable (for example, Windows without Triton), training falls back to the eager step automatically. Disable with `train.compile_cfc=false`.

### Configs

| Config | Backbone | Params | Role |
|---|---|---|---|
| `configs/libritts_r_cfc.yaml` | CfC, 12 blocks | 69.9M | main |
| `configs/libritts_r_transformer.yaml` | causal Transformer, 14 blocks | 70.9M | reference |
| `configs/ljspeech_cfc.yaml` | CfC, 12 blocks | 69.3M | single speaker |
| `configs/ljspeech_transformer.yaml` | causal Transformer, 14 blocks | 70.3M | single speaker |
| `configs/ljspeech_hybrid.yaml` | CfC + local attention (8 s window), 12 blocks | 67.2M | hybrid |
| `configs/ljspeech_lstm.yaml` | LSTM, 11 blocks | 71.4M | control |
| `configs/debug.yaml` | tiny CfC + local attention | 2.2M | smoke test |

All backbones are matched within ±5% of the CfC backbone. Any config value can be overridden from the command line, for example `train.lr=1e-4`.

## Repository layout

```
configs/              experiment configs (one file per run)
src/aqyn/
  cli.py              `aqyn` command: prepare / train / synth / eval / bench
  prepare.py          download LJSpeech / LibriTTS-R, resample, encode with Mimi, align words
  align.py            CTC forced alignment of words (wav2vec2 / MMS)
  train.py            training loop (graceful stop, exact resume)
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

- [x] Data: LJSpeech and LibriTTS-R download, Mimi tokenization, CTC word alignment
- [x] Stage 1 model: text window + word pointer, acoustic delay, control and stop heads
- [x] Training, synthesis, evaluation (ASR WER/CER, UTMOS, latency, Mimi ceiling), speed benchmark
- [x] Stage 1: first trained models on multi-speaker English (LibriTTS-R): CfC vs causal Transformer
- [ ] Stage 1: remaining backbones (LSTM, hybrid, Mamba2), second seed, UTMOS and speaker similarity
- [ ] Stage 1: overfitting of the stop / advance heads; checkpoint selection by codes loss
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
