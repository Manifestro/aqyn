import pytest
import torch

from aqyn.config import ModelConfig, TrainConfig
from aqyn.data import FrameBudgetSampler
from aqyn.latency import VOCAB as LAT_VOCAB
from aqyn.latency import stream
from aqyn.models import TTSModel, count_parameters
from aqyn.models.cfc import CfC

VOCAB = 40
PAD = 0


def tiny_cfg(layers, **kw) -> ModelConfig:
    base = {
        "codebook_size": 64,
        "text_dim": 32,
        "text_layers": 1,
        "dim": 32,
        "layers": layers,
        "heads": 2,
        "local_window": 4,
        "cfc_backbone_units": 32,
        "depth_dim": 16,
        "depth_layers": 1,
        "depth_heads": 2,
        "text_window": 8,
        "text_left": 2,
        "cross_attn_every": 2,
        "dropout": 0.0,
    }
    base.update(kw)
    return ModelConfig(**base)


def make_batch(v=64, k=8):
    """Three utterances with 4/3/2 words, 12/8/5 frames."""
    torch.manual_seed(0)
    word_starts = [[0, 4, 9, 13], [0, 3, 7], [0, 5]]
    text_lens = [17, 10, 8]
    word_frames = [[0, 3, 6, 9], [1, 2, 5], [0, 3]]
    code_lens = torch.tensor([12, 8, 5])
    b, n, w, t = 3, max(text_lens), 4, int(code_lens.max())
    text = torch.randint(4, VOCAB, (b, n))
    for i, ln in enumerate(text_lens):
        text[i, ln:] = PAD
    ws = torch.zeros(b, w, dtype=torch.long)
    wf = torch.full((b, w), 1 << 30, dtype=torch.long)
    for i in range(b):
        ws[i, : len(word_starts[i])] = torch.tensor(word_starts[i])
        wf[i, : len(word_frames[i])] = torch.tensor(word_frames[i])
    return {
        "text": text,
        "text_lens": torch.tensor(text_lens),
        "word_starts": ws,
        "word_frames": wf,
        "num_words": torch.tensor([4, 3, 2]),
        "codes": torch.randint(0, v, (b, t, k)),
        "code_lens": code_lens,
    }


@pytest.mark.parametrize("mode", ["default", "no_gate", "pure"])
def test_cfc_sequence_matches_steps(mode):
    torch.manual_seed(0)
    cfc = CfC(8, 16, backbone_units=16, backbone_layers=2, mode=mode)
    x = torch.randn(2, 10, 8)
    y, h_last = cfc(x)
    h = cfc.initial_state(2)
    for i in range(10):
        h = cfc.step(x[:, i], h)
        torch.testing.assert_close(h, y[:, i])
    torch.testing.assert_close(h, h_last)


def test_delay_roundtrip():
    model = TTSModel(tiny_cfg(["cfc"]), VOCAB, PAD)
    codes = torch.randint(0, 64, (1, 6, 8))
    rows = model.delay(codes)
    assert rows.shape == (1, 7, 8)
    assert (rows[0, 0, 1:] == model.bos).all() and rows[0, 6, 0] == model.bos
    torch.testing.assert_close(model.undelay(rows[0]), codes[0])


def test_teacher_pointers():
    model = TTSModel(tiny_cfg(["cfc"]), VOCAB, PAD)
    pointer, adv = model.pointers(make_batch())
    # utterance 0: words start at frames 0, 3, 6, 9 (12 frames + 1 delay row)
    assert pointer[0].tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
    assert adv[0].tolist() == [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
    # utterance 1: first word starts at frame 1, pointer stays clamped to word 0 before it
    assert pointer[1, :4].tolist() == [0, 0, 1, 1]
    # pointer never exceeds the last word
    assert (pointer[2] <= 1).all()


@pytest.mark.parametrize(
    "layers,pos",
    [
        (["cfc"] * 3, False),
        (["lstm"] * 2, False),
        (["attn"] * 2, True),
        (["cfc", "cfc", "local"], False),
    ],
)
def test_streaming_matches_teacher_forcing(layers, pos):
    torch.manual_seed(0)
    model = TTSModel(tiny_cfg(layers, audio_pos_emb=pos), VOCAB, PAD).eval()
    batch = make_batch()
    rows = model.delay(batch["codes"])
    pointer, _ = model.pointers(batch)
    with torch.no_grad():
        h_par = model.backbone(batch, rows, pointer)
        state = model.start(
            batch["text"], batch["text_lens"], batch["word_starts"], batch["num_words"]
        )
        prev = None
        for s in range(rows.shape[1]):
            state.pointer = pointer[:, s]
            h, _, _ = model.step(prev, state)
            torch.testing.assert_close(h, h_par[:, s], atol=1e-4, rtol=1e-4)
            prev = rows[:, s]


@pytest.mark.parametrize("layers", [["cfc"] * 2, ["cfc", "local"]])
def test_loss_backward(layers):
    model = TTSModel(tiny_cfg(layers), VOCAB, PAD)
    out = model(make_batch(), TrainConfig())
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, missing


def test_generate_shapes():
    model = TTSModel(tiny_cfg(["cfc", "local"]), VOCAB, PAD).eval()
    text = torch.randint(4, VOCAB, (12,))
    codes = model.generate(text, torch.tensor([0, 5, 9]), max_frames=6, stop_threshold=2.0)
    assert codes.shape == (6, 8)
    assert codes.max() < 64


def test_default_config_size():
    model = TTSModel(ModelConfig(), vocab_size=40, pad_id=0)
    total = count_parameters(model)["total"]
    assert 50e6 < total < 120e6, total


class _Lengths:
    def __init__(self, lengths):
        self.lengths = lengths

    def __len__(self):
        return len(self.lengths)

    def frames(self, i):
        return self.lengths[i]


def test_sampler_skip_resumes_mid_epoch():
    ds = _Lengths([20 + (i * 37) % 180 for i in range(300)])
    full = FrameBudgetSampler(ds, 1000, seed=3)
    full.set_epoch(2)
    resumed = FrameBudgetSampler(ds, 1000, seed=3)
    resumed.set_epoch(2, skip=5)
    assert list(resumed) == list(full)[5:]
    assert len(resumed) == len(full)
    resumed.set_epoch(3)
    assert len(list(resumed)) == len(resumed)


def test_stream_state_size():
    """A recurrent backbone keeps a fixed state; attention grows with the stream."""
    sizes = {}
    for kind in ("cfc", "attn"):
        model = TTSModel(tiny_cfg([kind, kind], audio_pos_emb=kind == "attn"), LAT_VOCAB, 0).eval()
        sizes[kind] = [stream(model, n, torch.device("cpu"))["state_bytes"] for n in (10, 30)]
    assert sizes["cfc"][0] == sizes["cfc"][1] > 0
    assert sizes["attn"][1] == 3 * sizes["attn"][0]
