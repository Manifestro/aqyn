import pytest
import torch

from aqyn.config import ModelConfig, TrainConfig
from aqyn.models import TTSModel, count_parameters
from aqyn.models.cfc import CfC

VOCAB = 40
PAD = 0


def tiny_cfg(layers, **kw) -> ModelConfig:
    base = dict(
        codebook_size=64,
        text_dim=32,
        text_layers=1,
        text_heads=2,
        dim=32,
        layers=layers,
        heads=2,
        local_window=4,
        cfc_backbone_units=32,
        depth_dim=16,
        depth_layers=1,
        depth_heads=2,
        dropout=0.0,
    )
    base.update(kw)
    return ModelConfig(**base)


def make_batch(b=3, t=12, n=9, k=8, v=64):
    torch.manual_seed(0)
    text_lens = torch.tensor([n, n - 3, n - 5])[:b]
    code_lens = torch.tensor([t, t - 4, t - 7])[:b]
    text = torch.randint(4, VOCAB, (b, n))
    text[torch.arange(n)[None] >= text_lens[:, None]] = PAD
    codes = torch.randint(0, v, (b, t, k))
    return {"text": text, "text_lens": text_lens, "codes": codes, "code_lens": code_lens}


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
    with torch.no_grad():
        h_par, _, _ = model.backbone(batch)
        state = model.start(batch["text"], batch["text_lens"])
        prev = None
        for t in range(batch["codes"].shape[1]):
            h, _, _ = model.step(prev, state)
            torch.testing.assert_close(h, h_par[:, t], atol=1e-4, rtol=1e-4)
            prev = batch["codes"][:, t]


@pytest.mark.parametrize("layers", [["cfc"] * 2, ["cfc", "local"]])
def test_loss_backward(layers):
    model = TTSModel(tiny_cfg(layers), VOCAB, PAD)
    out = model(make_batch(), TrainConfig())
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)


def test_generate_shapes():
    model = TTSModel(tiny_cfg(["cfc", "local"]), VOCAB, PAD).eval()
    codes = model.generate(torch.randint(4, VOCAB, (7,)), max_frames=6, min_frames=6)
    assert codes.shape == (6, 8)
    assert codes.max() < 64


def test_default_config_size():
    model = TTSModel(ModelConfig(), vocab_size=40, pad_id=0)
    total = count_parameters(model)["total"]
    assert 50e6 < total < 100e6, total
