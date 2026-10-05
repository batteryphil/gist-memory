"""
decay.py — Multi-Scale and Data-Dependent Decay Dynamics for Gist Memory
========================================================================
Provides flexible recency and forgetting mechanics for associative memory manifolds.
Supports:
1. Static scalar decay (default: 0.9995)
2. Learnable multi-scale decay: lambda_k = exp(-softplus(alpha_k))
3. Input-dependent / contextual decay: lambda_t = exp(-softplus(W x_t + b))
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple


class MultiScaleDecay(nn.Module):
    """
    Learnable multi-scale decay parameterization.
    
    Rather than constraining all memory channels to decay at the exact same rate,
    MultiScaleDecay initializes decay rates along a geometric progression of half-lives:
        half_life_k = min_half_life * (max_half_life / min_half_life) ** (k / (num_channels - 1))
    
    Parameterized as:
        decay_k = exp(-softplus(alpha_k))
    ensuring decay_k in (0, 1) strictly, preventing numerical explosion while allowing
    gradients to tune retention timescales per channel or head.
    """
    def __init__(
        self,
        num_channels: int,
        min_half_life: float = 8.0,
        max_half_life: float = 65536.0,
        learnable: bool = True,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.learnable = learnable

        if num_channels == 1:
            half_lives = torch.tensor([math.sqrt(min_half_life * max_half_life)], dtype=torch.float32)
        else:
            steps = torch.linspace(0, 1, num_channels, dtype=torch.float32)
            half_lives = min_half_life * ((max_half_life / min_half_life) ** steps)

        # target_decay = 0.5 ** (1.0 / half_life) = exp(-ln(2) / half_life)
        # target_rate = ln(2) / half_life
        # softplus(alpha) = target_rate => alpha = inverse_softplus(target_rate)
        target_rates = math.log(2.0) / half_lives
        # inverse_softplus(y) = ln(exp(y) - 1)
        # For small y: ln(y) is approximately true, but exact:
        init_alphas = torch.log(torch.expm1(target_rates))

        if learnable:
            self.raw_alpha = nn.Parameter(init_alphas)
        else:
            self.register_buffer("raw_alpha", init_alphas)

    def forward(self) -> torch.Tensor:
        """Returns decay rates [num_channels] strictly in (0, 1)."""
        rate = F.softplus(self.raw_alpha)
        decay = torch.exp(-rate)
        return decay

    def get_half_lives(self) -> torch.Tensor:
        """Returns the current effective half-life in tokens for each channel."""
        rates = F.softplus(self.raw_alpha)
        return math.log(2.0) / rates.clamp(min=1e-8)


class DataDependentDecay(nn.Module):
    """
    Data-dependent / contextual decay module.
    Computes token-wise decay rates:
        decay_t = exp(-softplus(W x_t + b))
    allowing the model to dynamically flush memory or accelerate consolidation
    during semantic task boundaries.
    """
    def __init__(
        self,
        d_model: int,
        num_channels: int,
        base_half_life: float = 2048.0,
    ):
        super().__init__()
        self.proj = nn.Linear(d_model, num_channels, bias=True)
        # Initialize bias to match base_half_life
        target_rate = math.log(2.0) / base_half_life
        init_bias = math.log(math.expm1(target_rate))
        nn.init.constant_(self.proj.bias, init_bias)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Hidden states [B, L, d_model]
        Returns:
            decay: Decay factors [B, L, num_channels] strictly in (0, 1)
        """
        raw = self.proj(x)
        rate = F.softplus(raw)
        return torch.exp(-rate)
