"""
multihead.py — Multi-Head Gist Memory Layer (MultiHeadGistLayer)
===============================================================
Disentangles associative memory into H independent heads:
  - Each head tracks specialized semantic invariants (entities, numerical values, logic, syntax).
  - Each head can possess its own time-scale / decay rate lambda_h.
  - Multi-head manifold state: M in R^(B x H x d_map x d_v), Z in R^(B x H x d_map).
  - Enables rich associative capacity without expanding per-head matrix dimensions.
"""

from __future__ import annotations
import math
from typing import Optional, Tuple, Union, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import RMSNorm
from .decay import MultiScaleDecay
from .state import GistState


class MultiHeadGistLayer(nn.Module):
    """
    Multi-Head Generative Thought Reconstruction Layer.
    
    Args:
        d_model: Hidden dimension of the host model (D).
        num_heads: Number of associative memory heads (H).
        d_map: Dimension of compressed blueprint per head (d_m, default: 16).
        d_v: Value dimension per head (default: d_model // num_heads).
        min_half_life: Shortest decay half-life in tokens for the head spectrum (default: 8.0).
        max_half_life: Longest decay half-life in tokens for the head spectrum (default: 65536.0).
        learnable_decay: Whether decay rates are trainable parameters.
        eps: Denominator stabilization constant.
        kernel: Positive feature map ("square", "relu2", "elu1").
        use_salience_gate: Whether to modulate accumulation with learnable novelty gates.
        fp32_accumulator: Whether state (M, Z) is accumulated in float32.
    """
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_map: int = 16,
        d_v: Optional[int] = None,
        min_half_life: float = 8.0,
        max_half_life: float = 65536.0,
        learnable_decay: bool = True,
        eps: float = 1e-4,
        kernel: Literal["square", "relu2", "elu1"] = "square",
        use_salience_gate: bool = True,
        fp32_accumulator: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_map = d_map
        self.d_v = d_v if d_v is not None else (d_model // num_heads)
        self.eps = eps
        self.kernel = kernel
        self.use_salience_gate = use_salience_gate
        self.fp32_accumulator = fp32_accumulator

        # 1. Multi-scale decay across heads
        self.decay = MultiScaleDecay(
            num_channels=num_heads,
            min_half_life=min_half_life,
            max_half_life=max_half_life,
            learnable=learnable_decay,
        )

        # 2. Multi-head blueprint encoder
        self.map_proj = nn.Linear(d_model, num_heads * d_map, bias=False)
        self.map_norm = RMSNorm(d_map)

        # 3. Multi-head query projection
        self.q_proj = nn.Linear(d_model, num_heads * d_map, bias=False)
        self.q_norm = RMSNorm(d_map)

        # 4. Multi-head value projection
        self.v_proj = nn.Linear(d_model, num_heads * self.d_v, bias=False)

        # 5. Salience / novelty gate per head
        if self.use_salience_gate:
            self.salience_gate = nn.Linear(d_model, num_heads, bias=True)
            nn.init.constant_(self.salience_gate.bias, 0.0)

        # 6. Reconstructive output projection & injection gate (Zero-Initialized)
        self.recon_proj = nn.Linear(num_heads * self.d_v, d_model, bias=False)
        self.recon_gate = nn.Linear(d_model, d_model, bias=True)
        nn.init.zeros_(self.recon_proj.weight)
        nn.init.constant_(self.recon_gate.bias, -2.0)

    @property
    def state_bytes(self) -> int:
        """Memory footprint of single-batch state across all heads in bytes."""
        return (self.num_heads * self.d_map * self.d_v + self.num_heads * self.d_map) * 4

    def _apply_kernel(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel == "square":
            return x * x
        elif self.kernel == "relu2":
            return F.relu(x) ** 2
        elif self.kernel == "elu1":
            return F.elu(x) + 1.0
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Union[GistState, Tuple[torch.Tensor, torch.Tensor]]] = None,
        return_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Union[GistState, Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Args:
            x: Input hidden states [B, L, D]
            state: Prior GistState or (M, Z) tuple [B, H, d_map, d_v], [B, H, d_map]
            return_state: Whether to return updated multi-head state
        """
        B, L, D = x.shape
        H = self.num_heads
        d_m = self.d_map
        d_v = self.d_v
        in_dtype = x.dtype
        device = x.device

        # Unpack state
        passed_as_obj = isinstance(state, GistState)
        prior_M, prior_Z, prior_steps = None, None, 0
        if state is not None:
            if passed_as_obj:
                prior_M, prior_Z = state.M, state.Z
                prior_steps = state.step_count
            else:
                prior_M, prior_Z = state
            if self.fp32_accumulator:
                prior_M = prior_M.float()
                prior_Z = prior_Z.float()

        # Step 1: Project & reshape into heads
        # m: [B, L, H, d_m]
        m = self.map_norm(self.map_proj(x).view(B, L, H, d_m))
        # q: [B, L, H, d_m]
        q = self.q_norm(self.q_proj(x).view(B, L, H, d_m))
        # v: [B, L, H, d_v]
        v = self.v_proj(x).view(B, L, H, d_v)

        m_k = self._apply_kernel(m)
        q_k = self._apply_kernel(q)

        if self.use_salience_gate:
            # gamma: [B, L, H, 1]
            gamma = torch.sigmoid(self.salience_gate(x)).unsqueeze(-1)
            m_k = m_k * gamma

        # Decays per head: [H] -> [1, 1, H, 1]
        decays = self.decay() # [H]

        # ─────────────────────────────────────────────────────────────────────
        # Branch A: Multi-Head Autoregressive Streaming Step (L == 1)
        # ─────────────────────────────────────────────────────────────────────
        if L == 1 and prior_M is not None and prior_Z is not None:
            # Squeeze time: [B, H, d_m], [B, H, d_v]
            m_t = m_k.squeeze(1)
            q_t = q_k.squeeze(1)
            v_t = v.squeeze(1)

            if self.fp32_accumulator:
                m_t = m_t.float()
                q_t = q_t.float()
                v_t = v_t.float()
                d_heads = decays.view(1, H, 1).float()
                d_heads_mat = decays.view(1, H, 1, 1).float()
            else:
                d_heads = decays.view(1, H, 1).to(in_dtype)
                d_heads_mat = decays.view(1, H, 1, 1).to(in_dtype)

            # Normalizer update: [B, H, d_m]
            new_Z = d_heads * prior_Z + m_t

            # Outer product update: [B, H, d_m, 1] @ [B, H, 1, d_v] = [B, H, d_m, d_v]
            outer_prod = torch.matmul(m_t.unsqueeze(-1), v_t.unsqueeze(-2))
            new_M = d_heads_mat * prior_M + outer_prod

            # Readout: [B, H, 1, d_m] @ [B, H, d_m, d_v] = [B, H, 1, d_v]
            num = torch.matmul(q_t.unsqueeze(-2), new_M)
            # den: [B, H, 1, d_m] @ [B, H, d_m, 1] = [B, H, 1, 1]
            den = torch.matmul(q_t.unsqueeze(-2), new_Z.unsqueeze(-1)) + self.eps

            recall_heads = (num / den.clamp(min=self.eps)).squeeze(-2).to(in_dtype) # [B, H, d_v]
            recall_flat = recall_heads.reshape(B, 1, H * d_v)

            recon = self.recon_proj(recall_flat)
            gate = torch.sigmoid(self.recon_gate(x))
            out = x + gate * recon

            new_state = None
            if return_state:
                if passed_as_obj or not isinstance(state, tuple):
                    new_state = GistState(
                        M=new_M,
                        Z=new_Z,
                        step_count=prior_steps + 1,
                        layer_idx=state.layer_idx if passed_as_obj else None,
                    )
                else:
                    new_state = (new_M, new_Z)

            return out, new_state

        # ─────────────────────────────────────────────────────────────────────
        # Branch B: Multi-Head Parallel Mode
        # ─────────────────────────────────────────────────────────────────────
        else:
            # Reshape for multi-head causal attention:
            # q_k: [B, H, L, d_m], m_k: [B, H, L, d_m], v: [B, H, L, d_v]
            q_h = q_k.permute(0, 2, 1, 3) # [B, H, L, d_m]
            m_h = m_k.permute(0, 2, 1, 3) # [B, H, L, d_m]
            v_h = v.permute(0, 2, 1, 3)   # [B, H, L, d_v]

            # Similarity: [B, H, L, L]
            S = torch.matmul(q_h, m_h.transpose(-2, -1))

            idx = torch.arange(L, device=device)
            dist = (idx.unsqueeze(1) - idx.unsqueeze(0)).clamp(min=0).float() # [L, L]
            causal_mask = (idx.unsqueeze(1) >= idx.unsqueeze(0)).unsqueeze(0).unsqueeze(0) # [1, 1, L, L]

            # decays: [H] -> [1, H, 1, 1]
            d_pow = (decays.view(1, H, 1, 1) ** dist.view(1, 1, L, L)).to(in_dtype)
            decay_mat = torch.where(causal_mask, d_pow, torch.zeros_like(d_pow))

            attn_weights = S * decay_mat # [B, H, L, L]
            recall_heads = torch.matmul(attn_weights, v_h) # [B, H, L, d_v]
            norm_factor = torch.sum(attn_weights, dim=-1, keepdim=True) # [B, H, L, 1]

            # Incorporate prior state if provided
            if prior_M is not None and prior_Z is not None:
                # prior_M: [B, H, d_m, d_v], prior_Z: [B, H, d_m]
                t_decay = (decays.view(1, H, 1, 1) ** (idx.view(1, 1, L, 1) + 1.0)).to(in_dtype)
                # q_h: [B, H, L, d_m] -> [B, H, L, 1, d_m]
                # prior_M: [B, H, 1, d_m, d_v]
                prior_recall = torch.matmul(q_h.unsqueeze(-2), prior_M.unsqueeze(2).to(in_dtype)).squeeze(-2) * t_decay
                recall_heads = recall_heads + prior_recall

                t_decay_Z = (decays.view(1, H, 1, 1) ** (idx.view(1, 1, L, 1) + 1.0)).to(in_dtype)
                prior_norm = torch.matmul(q_h.unsqueeze(-2), prior_Z.unsqueeze(-1).unsqueeze(2).to(in_dtype)).squeeze(-2) * t_decay_Z
                norm_factor = norm_factor + prior_norm

            norm_factor = norm_factor + self.eps
            recall_heads = (recall_heads / norm_factor.clamp(min=self.eps)).to(in_dtype)

            # Transpose and flatten heads: [B, L, H * d_v]
            recall_flat = recall_heads.permute(0, 2, 1, 3).reshape(B, L, H * d_v)

            recon = self.recon_proj(recall_flat)
            gate = torch.sigmoid(self.recon_gate(x))
            out = x + gate * recon

            new_state = None
            if return_state:
                # Weights: [1, H, L, 1]
                decay_weights = (decays.view(1, H, 1, 1) ** (L - 1 - idx).view(1, 1, L, 1)).to(in_dtype)
                m_decayed = m_h * decay_weights # [B, H, L, d_m]

                m_acc = m_decayed.float() if self.fp32_accumulator else m_decayed
                v_acc = v_h.float() if self.fp32_accumulator else v_h

                M_new = torch.matmul(m_acc.transpose(-2, -1), v_acc) # [B, H, d_m, d_v]
                Z_new = torch.sum(m_acc, dim=2) # [B, H, d_m]

                if prior_M is not None and prior_Z is not None:
                    decay_L = (decays.view(1, H, 1, 1) ** L).float()
                    decay_L_Z = (decays.view(1, H, 1) ** L).float()
                    M_new = M_new + decay_L * prior_M
                    Z_new = Z_new + decay_L_Z * prior_Z

                if passed_as_obj or (state is None and not isinstance(state, tuple)):
                    new_state = GistState(
                        M=M_new,
                        Z=Z_new,
                        step_count=prior_steps + L,
                        layer_idx=state.layer_idx if passed_as_obj else None,
                    )
                else:
                    new_state = (M_new, Z_new)

            return out, new_state
