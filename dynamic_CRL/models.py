import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from sim import SimulationData
from utils import normalized_time_encoding


class RecordingContext(nn.Module):
    """Fixed-dimensional summary of a variable-length (Y, I) recording."""

    def __init__(self, cfg: Any):
        super().__init__()
        width = cfg.CONTEXT_WIDTH
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.P + cfg.A, width)
        self.summary = nn.Parameter(
            torch.randn(cfg.CONTEXT_TOKENS, width) / math.sqrt(width)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=cfg.CONTEXT_HEADS,
            dim_feedforward=2 * width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=cfg.CONTEXT_LAYERS
        )
        self.norm = nn.LayerNorm(width)

    def forward(
        self,
        y_padded: torch.Tensor,
        valid_mask: torch.Tensor,
        intervention: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_length, _ = y_padded.shape
        intervention_repeated = intervention[:, None, :].expand(-1, max_length, -1)
        tokens = self.input_proj(
            torch.cat([y_padded, intervention_repeated], dim=-1)
        )
        tokens = tokens + normalized_time_encoding(
            valid_mask, self.cfg.CONTEXT_WIDTH
        )

        summary = self.summary[None, :, :].expand(batch_size, -1, -1)
        tokens = torch.cat([summary, tokens], dim=1)
        padding = torch.cat(
            [
                torch.zeros(
                    batch_size,
                    self.cfg.CONTEXT_TOKENS,
                    dtype=torch.bool,
                    device=y_padded.device,
                ),
                ~valid_mask.bool(),
            ],
            dim=1,
        )
        encoded = self.transformer(tokens, src_key_padding_mask=padding)
        return self.norm(encoded[:, : self.cfg.CONTEXT_TOKENS]).reshape(
            batch_size, -1
        )


class ContextReconstructionDecoder(nn.Module):
    def __init__(self, cfg: Any):
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(cfg.CONTEXT_DIM + cfg.CONTEXT_WIDTH, 2 * cfg.CONTEXT_WIDTH),
            nn.GELU(),
            nn.Linear(2 * cfg.CONTEXT_WIDTH, 2 * cfg.CONTEXT_WIDTH),
            nn.GELU(),
            nn.Linear(2 * cfg.CONTEXT_WIDTH, cfg.P),
        )

    def forward(
        self, context: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        _, max_length = valid_mask.shape
        position = normalized_time_encoding(valid_mask, self.cfg.CONTEXT_WIDTH)
        repeated_context = context[:, None, :].expand(-1, max_length, -1)
        return self.net(torch.cat([repeated_context, position], dim=-1))


class CausalEncoder(nn.Module):
    """Direct Y -> causal slots, initialized outside the module."""

    def __init__(
        self,
        cfg: Any,
        initial_weight: np.ndarray,
        initial_bias: np.ndarray,
    ):
        super().__init__()
        self.cfg = cfg
        self.base = nn.Linear(cfg.P, cfg.D)
        with torch.no_grad():
            self.base.weight.copy_(
                torch.tensor(initial_weight, dtype=torch.float32)
            )
            self.base.bias.copy_(torch.tensor(initial_bias, dtype=torch.float32))
        self.residual = nn.Sequential(
            nn.Linear(cfg.P, 32),
            nn.SiLU(),
            nn.Linear(32, cfg.D),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.base(observations) + self.cfg.ENCODER_RESIDUAL_STRENGTH * self.residual(
            observations
        )


class ResidualDecoder(nn.Module):
    def __init__(
        self,
        cfg: Any,
        encoder: CausalEncoder,
        data: SimulationData,
    ):
        super().__init__()
        self.base = nn.Linear(cfg.D, cfg.P)
        with torch.no_grad():
            y_flat = data.Y_np[data.train_idx].reshape(-1, cfg.P)
            z_flat = encoder(
                torch.tensor(y_flat, dtype=torch.float32, device=data.device)
            ).cpu().numpy()
            design = np.c_[z_flat, np.ones(len(z_flat))]
            coefficients = np.linalg.lstsq(design, y_flat, rcond=None)[0]
            self.base.weight.copy_(
                torch.tensor(coefficients[:-1].T, dtype=torch.float32)
            )
            self.base.bias.copy_(
                torch.tensor(coefficients[-1], dtype=torch.float32)
            )
        self.residual = nn.Sequential(
            nn.Linear(cfg.D, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, cfg.P),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, latent_raw: torch.Tensor) -> torch.Tensor:
        return self.base(latent_raw) + 0.10 * self.residual(latent_raw)


class TemporalCoupledFM(nn.Module):
    def __init__(
        self,
        cfg: Any,
        e_lag: int,
        e_inst: int,
        h_dim: int,
    ):
        super().__init__()
        graph_input_dim = 2 * e_lag + 3 * e_inst
        input_dim = cfg.CONTEXT_DIM + graph_input_dim + h_dim + 1
        self.e_lag = e_lag
        self.e_inst = e_inst
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, cfg.FM_HIDDEN),
            nn.SiLU(),
            nn.Linear(cfg.FM_HIDDEN, cfg.FM_HIDDEN),
            nn.SiLU(),
            nn.Linear(cfg.FM_HIDDEN, cfg.FM_HIDDEN),
            nn.SiLU(),
        )
        self.lag_logits = nn.Linear(cfg.FM_HIDDEN, e_lag * 2)
        self.inst_logits = nn.Linear(cfg.FM_HIDDEN, e_inst * 3)
        self.h_velocity = nn.Linear(cfg.FM_HIDDEN, h_dim)

    def forward(
        self,
        context: torch.Tensor,
        lag_state: torch.Tensor,
        inst_state: torch.Tensor,
        h_state: torch.Tensor,
        time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lag_one_hot = torch.nn.functional.one_hot(
            lag_state.long(), 2
        ).float().flatten(1)
        inst_one_hot = torch.nn.functional.one_hot(
            inst_state.long(), 3
        ).float().flatten(1)
        hidden = self.trunk(
            torch.cat(
                [context, lag_one_hot, inst_one_hot, h_state, time], dim=1
            )
        )
        lag_logits = self.lag_logits(hidden).reshape(-1, self.e_lag, 2)
        inst_logits = self.inst_logits(hidden).reshape(-1, self.e_inst, 3)
        return lag_logits, inst_logits, self.h_velocity(hidden)
