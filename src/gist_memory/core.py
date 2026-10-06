"""
core.py — GistLayer: decayed kernelized linear attention with a gated residual
=============================================================================
Mathematically, GistLayer is linear attention with exponential decay — the same
family as Katharopoulos et al. 2020 ("Transformers are RNNs"), RetNet (Sun et al.
2023), and Gated Linear Attention (Yang et al. 2023):

    k_t = phi(RMSNorm(W_k x_t)) * gamma_t      (gamma_t: optional sigmoid gate)
    q_t = phi(RMSNorm(W_q x_t))
    v_t = W_v x_t
    M_t = lambda * M_{t-1} + k_t v_t^T         (d_map x D state)
    Z_t = lambda * Z_{t-1} + k_t
    y_t = (q_t^T M_t) / (q_t^T Z_t + eps)
    out = x_t + sigmoid(W_g x_t + b_g) * (W_o y_t)     (W_o zero-initialized)

In code: map_proj = W_k ("blueprint"), q_proj = W_q, v_proj = W_v,
recon_proj = W_o, recon_gate = W_g.

Complexity notes:
- Streaming decode (L == 1 with a prior state) is O(d_map * D) per token.
- The parallel path below materializes an [B, L, L] matrix, i.e. O(L^2) time
  and memory. Use ChunkedGistLayer for long sequences.
- A fixed-size state necessarily loses information as sequences grow; capacity
  is bounded by d_map. Nothing here has been trained or evaluated for quality.
"""

from __future__ import annotations
import math
from typing import Optional, Tuple, Union, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

from .decay import MultiScaleDecay, DataDependentDecay
from .state import GistState


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


class GistLayer(nn.Module):
    """
    Second-Order Generative Thought Reconstruction Layer.
    
    Args:
        d_model: Hidden dimension of the host transformer trunk (D).
        d_map: Dimension of the compressed topological blueprint (d_m, default: 32).
        decay: Exponential recency decay rate (scalar float, MultiScaleDecay, or None).
        eps: Denominator stabilization constant (default: 1e-4).
        kernel: Positive feature map: "square" (x^2), "relu2" (ReLU(x)^2), or "elu1" (ELU(x) + 1).
        use_salience_gate: Whether to modulate memory accumulation by learnable novelty gate.
        fp32_accumulator: Whether to maintain manifold state (M, Z) in float32 for maximum precision.
    """
    def __init__(
        self,
        d_model: int = 512,
        d_map: int = 32,
        decay: Union[float, nn.Module] = 0.9995,
        eps: float = 1e-4,
        kernel: Literal["square", "relu2", "elu1", "prime"] = "square",
        use_salience_gate: bool = True,
        fp32_accumulator: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_map = d_map
        self.eps = eps
        self.kernel = kernel
        self.use_salience_gate = use_salience_gate
        self.fp32_accumulator = fp32_accumulator

        # 1. Decay mechanism
        if isinstance(decay, (int, float)):
            self.register_buffer("decay", torch.tensor(float(decay), dtype=torch.float32))
            self.decay_module = None
        elif isinstance(decay, nn.Module):
            self.decay = None
            self.decay_module = decay
        else:
            raise ValueError(f"Unsupported decay specification: {decay}")

        # 2. Topological Blueprint Encoder (d_model -> d_map)
        self.map_proj = nn.Linear(d_model, d_map, bias=False)
        self.map_norm = RMSNorm(d_map)

        # 3. Novelty / Salience Gate
        if self.use_salience_gate:
            self.salience_gate = nn.Linear(d_model, 1, bias=True)
            nn.init.constant_(self.salience_gate.bias, 0.0) # Neutral initial gate

        # 4. Memory Value Projection (information payload)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # 5. Reconstruction Query Projection (d_model -> d_map)
        self.q_proj = nn.Linear(d_model, d_map, bias=False)
        self.q_norm = RMSNorm(d_map)

        # 6. Generative Thought Reconstruction & Residual Gating (Zero-Initialized)
        self.recon_proj = nn.Linear(d_model, d_model, bias=False)
        self.recon_gate = nn.Linear(d_model, d_model, bias=True)
        nn.init.zeros_(self.recon_proj.weight) # Strict zero-init: output is identity at step 0
        nn.init.constant_(self.recon_gate.bias, -2.0) # Starts gentle, learns injection magnitude

    @property
    def state_bytes(self) -> int:
        """Memory footprint of single-batch state in bytes (fp32)."""
        return (self.d_map * self.d_model + self.d_map) * 4

    def _apply_kernel(self, x: torch.Tensor) -> torch.Tensor:
        """Positive feature mapping to ensure non-negative kernel similarity."""
        if self.kernel == "square":
            return x * x
        elif self.kernel == "relu2":
            return F.relu(x) ** 2
        elif self.kernel == "elu1":
            return F.elu(x) + 1.0
        elif self.kernel == "prime":
            return torch.exp(4.0 * x)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def _get_decay(self, x: torch.Tensor) -> Union[torch.Tensor, float]:
        """Resolves current decay rate / factors."""
        if self.decay is not None:
            return self.decay
        elif isinstance(self.decay_module, MultiScaleDecay):
            return self.decay_module()
        elif isinstance(self.decay_module, DataDependentDecay):
            return self.decay_module(x)
        else:
            return 0.9995

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Union[GistState, Tuple[torch.Tensor, torch.Tensor]]] = None,
        return_state: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Union[GistState, Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass through Gist memory.
        
        Args:
            x: Input hidden states [B, L, D]
            state: Prior GistState or (M, Z) tuple from previous sequence/steps.
            return_state: Whether to compute and return the updated state.
            attention_mask: Optional binary or boolean mask [B, L] or [B, 1, 1, L].
            
        Returns:
            out: Reconstructed and gated hidden states [B, L, D]
            new_state: Updated GistState if return_state=True, else None
        """
        B, L, D = x.shape
        in_dtype = x.dtype
        device = x.device

        # Process attention mask if provided
        mask = None
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                mask = attention_mask.unsqueeze(-1).to(in_dtype)
            elif attention_mask.ndim == 3:
                mask = attention_mask.transpose(1, 2).to(in_dtype) if attention_mask.shape[1] == 1 else attention_mask.to(in_dtype)
            elif attention_mask.ndim == 4:
                mask = attention_mask[:, 0, 0, :].unsqueeze(-1).to(in_dtype) if (attention_mask.shape[1] == 1 and attention_mask.shape[2] == 1) else attention_mask[:, :, -1, :].unsqueeze(-1).to(in_dtype)
            if mask is not None and mask.shape[1] != L:
                mask = mask[:, -L:, :]
            if mask is not None and (mask < 0).any():
                mask = (mask >= -1.0).to(in_dtype)

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

        # Step 1: Distill topological thought blueprint m_t
        m_raw = self.map_proj(x)
        m = self.map_norm(m_raw) # [B, L, d_map]

        # Step 2: Query projection
        q_raw = self.q_proj(x)
        q = self.q_norm(q_raw) # [B, L, d_map]

        # Step 3: Value payload
        v = self.v_proj(x) # [B, L, D]

        # Step 4: Kernel mapping & salience gating
        m_k = self._apply_kernel(m)
        q_k = self._apply_kernel(q)

        if self.use_salience_gate:
            gamma = torch.sigmoid(self.salience_gate(x)) # [B, L, 1]
            m_k = m_k * gamma

        if mask is not None:
            m_k = m_k * mask
            v = v * mask

        decay_val = self._get_decay(x)

        # ─────────────────────────────────────────────────────────────────────
        # Branch A: Autoregressive Streaming Step (L == 1 and prior state exists)
        # ─────────────────────────────────────────────────────────────────────
        if L == 1 and prior_M is not None and prior_Z is not None:
            v_t = v.squeeze(1) # [B, D]
            m_t = m_k.squeeze(1) # [B, d_map]
            q_t = q_k.squeeze(1) # [B, d_map]

            if self.fp32_accumulator:
                v_t_acc = v_t.float()
                m_t_acc = m_t.float()
                q_t_acc = q_t.float()
                d_val = decay_val.float() if isinstance(decay_val, torch.Tensor) else float(decay_val)
            else:
                v_t_acc = v_t
                m_t_acc = m_t
                q_t_acc = q_t
                d_val = decay_val.to(v.dtype) if isinstance(decay_val, torch.Tensor) else decay_val

            # Ensure d_val broadcasts properly without triggering 3D batch broadcasting
            if isinstance(d_val, torch.Tensor):
                while d_val.ndim > 2:
                    d_val = d_val.squeeze(1)
                if d_val.ndim == 1 and d_val.shape[0] != B:
                    d_val = d_val.unsqueeze(0)
                elif d_val.ndim == 1 and d_val.shape[0] == B:
                    d_val = d_val.unsqueeze(1)
                elif d_val.ndim == 0:
                    d_val = d_val.item()

            if isinstance(d_val, torch.Tensor):
                decay_Z = d_val * prior_Z
                decay_M = d_val.unsqueeze(-1) * prior_M
            else:
                decay_Z = d_val * prior_Z
                decay_M = d_val * prior_M

            # State recurrence
            outer_prod = torch.bmm(m_t_acc.unsqueeze(2), v_t_acc.unsqueeze(1)) # [B, d_map, D]
            if mask is not None:
                mask_t = mask.squeeze(1).float() if self.fp32_accumulator else mask.squeeze(1)
                new_Z = torch.where(mask_t > 0, decay_Z + m_t_acc, prior_Z)
                new_M = torch.where(mask_t.unsqueeze(-1) > 0, decay_M + outer_prod, prior_M)
            else:
                new_Z = decay_Z + m_t_acc
                new_M = decay_M + outer_prod

            # Readout: recall = q_t^T M / (q_t^T Z + eps)
            num = torch.bmm(q_t_acc.unsqueeze(1), new_M) # [B, 1, D]
            den = torch.bmm(q_t_acc.unsqueeze(1), new_Z.unsqueeze(-1)) + self.eps # [B, 1, 1]
            recall = (num / den.clamp(min=self.eps)).to(in_dtype) # [B, 1, D]

            # Reconstruct thought and residual injection
            recon = self.recon_proj(recall)
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
        # Branch B: Parallel Causal Training Mode
        # ─────────────────────────────────────────────────────────────────────
        else:
            # S[b, i, j] = q_i^T m_j
            S = torch.bmm(q_k, m_k.transpose(1, 2)) # [B, L, L]

            idx = torch.arange(L, device=device)
            causal_mask = idx.unsqueeze(1) >= idx.unsqueeze(0) # [L, L]

            is_seq_decay = isinstance(decay_val, torch.Tensor) and decay_val.ndim >= 2 and decay_val.shape[1] == L

            if is_seq_decay:
                decay_seq = decay_val.float()
                if decay_seq.ndim == 3 and decay_seq.shape[-1] == 1:
                    decay_seq = decay_seq.squeeze(-1) # [B, L]
                log_decay = torch.log(decay_seq.clamp(min=1e-8, max=1.0)) # [B, L]
                cum_log = torch.cumsum(log_decay, dim=1) # [B, L]
                dist_cum = (cum_log.unsqueeze(2) - cum_log.unsqueeze(1)).clamp(max=0.0) # [B, L, L]
                decay_mat = torch.where(
                    causal_mask,
                    torch.exp(dist_cum).to(in_dtype),
                    torch.zeros_like(dist_cum, dtype=in_dtype)
                ) # [B, L, L]
                t_decay = torch.exp(cum_log).view(B, L, 1, 1).to(in_dtype)
                t_decay_Z = torch.exp(cum_log).view(B, L, 1).to(in_dtype)
                end_cum = cum_log[:, -1:].unsqueeze(1)
                boundary_weights = torch.exp(end_cum - cum_log.unsqueeze(1)).to(in_dtype).transpose(1, 2) # [B, L, 1]
                decay_L = torch.exp(cum_log[:, -1:]).view(B, 1, 1).float()
            else:
                if isinstance(decay_val, torch.Tensor) and decay_val.numel() == 1:
                    decay_scalar = decay_val.item()
                elif isinstance(decay_val, (int, float)):
                    decay_scalar = float(decay_val)
                else:
                    decay_scalar = float(decay_val.mean().item())

                dist = idx.unsqueeze(1) - idx.unsqueeze(0)
                decay_mat = torch.where(
                    causal_mask,
                    (decay_scalar ** dist.float()).to(in_dtype),
                    torch.zeros_like(dist, dtype=in_dtype)
                ).unsqueeze(0) # [1, L, L]
                t_decay = (decay_scalar ** (idx.float() + 1.0)).view(1, L, 1, 1).to(in_dtype)
                t_decay_Z = (decay_scalar ** (idx.float() + 1.0)).view(1, L, 1).to(in_dtype)
                boundary_weights = (decay_scalar ** (L - 1 - idx).float()).view(1, L, 1).to(in_dtype)
                decay_L = (decay_scalar ** L)

            attn_weights = S * decay_mat # [B, L, L]
            if mask is not None:
                attn_weights = attn_weights * mask.transpose(1, 2)

            recall_all = torch.bmm(attn_weights, v) # [B, L, D]
            norm_factor = torch.sum(attn_weights, dim=-1, keepdim=True) # [B, L, 1]

            # Incorporate prior state if provided (chunked / recurrent prefix)
            if prior_M is not None and prior_Z is not None:
                prior_recall = torch.matmul(q_k.unsqueeze(2), prior_M.unsqueeze(1).to(in_dtype)) * t_decay
                recall_all = recall_all + prior_recall.squeeze(2)

                prior_norm = torch.bmm(q_k, prior_Z.unsqueeze(-1).to(in_dtype)) * t_decay_Z
                norm_factor = norm_factor + prior_norm

            norm_factor = norm_factor + self.eps
            recall_all = (recall_all / norm_factor.clamp(min=self.eps)).to(in_dtype)

            recon = self.recon_proj(recall_all)
            gate = torch.sigmoid(self.recon_gate(x))
            out = x + gate * recon

            new_state = None
            if return_state:
                # Cumulative state at sequence boundary
                m_decayed = m_k * boundary_weights # [B, L, d_map]

                v_acc = v.float() if self.fp32_accumulator else v
                m_acc = m_decayed.float() if self.fp32_accumulator else m_decayed

                M_new = torch.bmm(m_acc.transpose(1, 2), v_acc) # [B, d_map, D]
                Z_new = torch.sum(m_acc, dim=1) # [B, d_map]

                if prior_M is not None and prior_Z is not None:
                    M_new = M_new + decay_L * prior_M
                    decay_L_Z = decay_L.squeeze(-1) if isinstance(decay_L, torch.Tensor) else decay_L
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


# Alias for seamless drop-in backwards compatibility with PRIME-Moment-Attention
GenerativeThoughtReconstructionLayer = GistLayer
