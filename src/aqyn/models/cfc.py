"""Closed-form Continuous-time (CfC) recurrent cell and layer.

Follows Hasani et al., "Closed-form continuous-time neural networks" (2022):

    z      = backbone([x, h])
    f1, f2 = tanh(W1 z), tanh(W2 z)
    gate   = sigmoid(Wa z * dt + Wb z)
    h'     = f1 * (1 - gate) + gate * f2          (mode "default")
    h'     = f1 + gate * f2                       (mode "no_gate")
    h'     = -A * exp(-dt * (|w_tau| + |f1|)) * f1 + A   (mode "pure")

The input part of the first backbone layer is applied to the whole sequence at once;
only the recurrent part runs inside the time loop.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F
from torch import nn

MODES = ("default", "no_gate", "pure")


def lecun_tanh(x: torch.Tensor) -> torch.Tensor:
    return 1.7159 * torch.tanh(0.666 * x)


class CfCCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        backbone_units: int = 512,
        backbone_layers: int = 1,
        mode: str = "default",
        dropout: float = 0.0,
    ):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if backbone_layers < 1:
            raise ValueError("backbone_layers must be >= 1")
        self.hidden_size = hidden_size
        self.mode = mode
        self.in_proj = nn.Linear(input_size, backbone_units)
        self.rec_proj = nn.Linear(hidden_size, backbone_units, bias=False)
        self.backbone = nn.ModuleList(
            nn.Linear(backbone_units, backbone_units) for _ in range(backbone_layers - 1)
        )
        self.dropout = nn.Dropout(dropout)
        if mode == "pure":
            self.heads = nn.Linear(backbone_units, hidden_size)
            self.w_tau = nn.Parameter(torch.zeros(hidden_size))
            self.A = nn.Parameter(torch.ones(hidden_size))
        else:
            # ff1, ff2, time_a, time_b fused into one matmul
            self.heads = nn.Linear(backbone_units, 4 * hidden_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.orthogonal_(self.rec_proj.weight)
        for layer in self.backbone:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.heads.weight)
        nn.init.zeros_(self.heads.bias)

    def project_input(self, x: torch.Tensor) -> torch.Tensor:
        """Input part of the first backbone layer, for any leading shape."""
        return self.in_proj(x)

    def step(self, x_proj: torch.Tensor, h: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        backbone = tuple((layer.weight, layer.bias) for layer in self.backbone)
        pure = self.mode == "pure"
        return _STEP[0](
            x_proj,
            h,
            self.rec_proj.weight,
            backbone,
            self.heads.weight,
            self.heads.bias,
            self.w_tau if pure else None,
            self.A if pure else None,
            self.mode,
            dt,
            self.dropout.p,
            self.training,
        )


def cfc_step(
    x_proj: torch.Tensor,
    h: torch.Tensor,
    rec_w: torch.Tensor,
    backbone: tuple,
    heads_w: torch.Tensor,
    heads_b: torch.Tensor,
    w_tau: torch.Tensor | None,
    A: torch.Tensor | None,
    mode: str,
    dt: float,
    p_drop: float,
    training: bool,
) -> torch.Tensor:
    """One CfC update as a pure function, so a single compiled graph serves every layer."""
    z = lecun_tanh(x_proj + F.linear(h, rec_w))
    for w, b in backbone:
        z = lecun_tanh(F.linear(z, w, b))
    z = F.dropout(z, p_drop, training)
    if mode == "pure":
        f1 = F.linear(z, heads_w, heads_b)
        return -A * torch.exp(-dt * (w_tau.abs() + f1.abs())) * f1 + A
    f1, f2, ta, tb = F.linear(z, heads_w, heads_b).chunk(4, dim=-1)
    f1, f2 = torch.tanh(f1), torch.tanh(f2)
    gate = torch.sigmoid(ta * dt + tb)
    if mode == "no_gate":
        return f1 + gate * f2
    return f1 * (1.0 - gate) + gate * f2


# The step implementation in use; enable_compiled_step() swaps in a compiled version.
_STEP = [cfc_step]


def enable_compiled_step(device: torch.device, autocast=None) -> bool:
    """Compile the CfC step with ``torch.compile`` so its ~15 small kernels fuse into a few.

    Runs a short self-test (train and inference, with and without autocast). If anything
    fails, for example no Triton on this platform, it stays on the eager version and
    returns False.
    """
    try:
        compiled = torch.compile(cfc_step, dynamic=True)
        d = 64
        lin = nn.Linear(d, 4 * d).to(device)
        rec = torch.randn(d, d, device=device, requires_grad=True)
        contexts = [contextlib.nullcontext]
        if autocast is not None:
            contexts.append(autocast)
        for ctx in contexts:
            for b in (3, 5):  # two batch sizes, to settle dynamic shapes
                x = torch.randn(b, d, device=device)
                h = torch.zeros(b, d, device=device)
                with ctx():
                    y = compiled(
                        x, h, rec, (), lin.weight, lin.bias, None, None, "default", 1.0, 0.1, True
                    )
                y.float().sum().backward()
                with torch.no_grad(), ctx():
                    compiled(
                        x, h, rec, (), lin.weight, lin.bias, None, None, "default", 1.0, 0.0, False
                    )
    except Exception as e:  # noqa: BLE001 - any compiler failure means "use eager"
        print(f"[cfc] torch.compile unavailable, using eager CfC step: {type(e).__name__}: {e}")
        _STEP[0] = cfc_step
        return False
    _STEP[0] = compiled
    return True


def disable_compiled_step() -> None:
    _STEP[0] = cfc_step


class CfC(nn.Module):
    """Sequence wrapper with a learned initial state. Input/output are ``[B, T, D]``."""

    def __init__(self, input_size: int, hidden_size: int, **cell_kwargs):
        super().__init__()
        self.cell = CfCCell(input_size, hidden_size, **cell_kwargs)
        self.h0 = nn.Parameter(torch.zeros(hidden_size))

    def initial_state(self, batch: int) -> torch.Tensor:
        return self.h0.unsqueeze(0).expand(batch, -1)

    def forward(
        self, x: torch.Tensor, h: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, _ = x.shape
        if h is None:
            h = self.initial_state(b)
        xp = self.cell.project_input(x)
        outs = []
        for i in range(t):
            h = self.cell.step(xp[:, i], h)
            outs.append(h)
        return torch.stack(outs, dim=1), h

    def step(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """One frame: ``x`` is ``[B, D]``; returns the new state (which is also the output)."""
        return self.cell.step(self.cell.project_input(x), h)
