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

import torch
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
        z = lecun_tanh(x_proj + self.rec_proj(h))
        for layer in self.backbone:
            z = lecun_tanh(layer(z))
        z = self.dropout(z)
        if self.mode == "pure":
            f1 = self.heads(z)
            return -self.A * torch.exp(-dt * (self.w_tau.abs() + f1.abs())) * f1 + self.A
        f1, f2, ta, tb = self.heads(z).chunk(4, dim=-1)
        f1, f2 = torch.tanh(f1), torch.tanh(f2)
        gate = torch.sigmoid(ta * dt + tb)
        if self.mode == "no_gate":
            return f1 + gate * f2
        return f1 * (1.0 - gate) + gate * f2


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
