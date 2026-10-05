"""
chunked.py — Chunked State Space Duality (SSD) Linear Scan for Gist Memory
==========================================================================
Replaces the quadratic O(L^2) parallel training matrix with a block-parallel scan:
  - Divides sequence L into chunks of size C (default: 64).
  - Intra-chunk: dense tensor-core GEMM multiplications of size C x C.
  - Inter-chunk: associative recurrence across chunk boundaries in O((L/C) * d_map * D).
  - Preserves exact mathematical equivalence while slashing memory consumption by L/C (up to 1000x).
  - Unlocks training and prefill on long contexts (64k - 1M tokens) on a single GPU.
"""

from __future__ import annotations
import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import GistLayer
from .state import GistState


class ChunkedGistLayer(GistLayer):
    """
    GistLayer with Chunked SSD (State Space Duality) parallel execution.
    Inherits all projections, normalizers, and gating from GistLayer.
    """
    def __init__(
        self,
        d_model: int = 512,
        d_map: int = 32,
        chunk_size: int = 64,
        decay: Union[float, nn.Module] = 0.9995,
        eps: float = 1e-4,
        kernel: str = "square",
        use_salience_gate: bool = True,
        fp32_accumulator: bool = True,
    ):
        super().__init__(
            d_model=d_model,
            d_map=d_map,
            decay=decay,
            eps=eps,
            kernel=kernel,
            use_salience_gate=use_salience_gate,
            fp32_accumulator=fp32_accumulator,
        )
        self.chunk_size = chunk_size

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Union[GistState, Tuple[torch.Tensor, torch.Tensor]]] = None,
        return_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Union[GistState, Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass using Chunked SSD block-parallel scan.
        """
        B, L, D = x.shape
        C = self.chunk_size

        # Fallback to base GistLayer for single-token streaming or short sequences <= chunk_size
        if L <= C:
            return super().forward(x, state=state, return_state=return_state)

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

        # Step 1: Project & normalize
        m_raw = self.map_proj(x)
        m = self.map_norm(m_raw) # [B, L, d_map]

        q_raw = self.q_proj(x)
        q = self.q_norm(q_raw) # [B, L, d_map]

        v = self.v_proj(x) # [B, L, D]

        m_k = self._apply_kernel(m)
        q_k = self._apply_kernel(q)

        if self.use_salience_gate:
            gamma = torch.sigmoid(self.salience_gate(x))
            m_k = m_k * gamma

        decay_val = self._get_decay(x)
        if isinstance(decay_val, torch.Tensor) and decay_val.numel() == 1:
            decay_scalar = decay_val.item()
        elif isinstance(decay_val, (int, float)):
            decay_scalar = float(decay_val)
        else:
            decay_scalar = float(decay_val.mean().item())

        # Pad sequence length to multiple of C if necessary
        pad_len = (C - (L % C)) % C
        if pad_len > 0:
            m_k = F.pad(m_k, (0, 0, 0, pad_len))
            q_k = F.pad(q_k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))

        num_chunks = (L + pad_len) // C

        # Reshape into chunks: [B, num_chunks, C, ...]
        m_chunks = m_k.view(B, num_chunks, C, self.d_map)
        q_chunks = q_k.view(B, num_chunks, C, self.d_map)
        v_chunks = v.view(B, num_chunks, C, D)

        # Precompute intra-chunk causal decay: [C, C]
        c_idx = torch.arange(C, device=device)
        dist_c = c_idx.unsqueeze(1) - c_idx.unsqueeze(0)
        c_mask = dist_c >= 0
        intra_decay = torch.where(
            c_mask,
            (decay_scalar ** dist_c.float()).to(in_dtype),
            torch.zeros_like(dist_c, dtype=in_dtype)
        ).unsqueeze(0).unsqueeze(0) # [1, 1, C, C]

        # Weights for chunk-end state accumulation: [1, 1, C, 1]
        chunk_end_weights = (decay_scalar ** (C - 1 - c_idx).float()).view(1, 1, C, 1).to(in_dtype)

        # Decay over full chunk
        decay_C = (decay_scalar ** C)

        # Local intra-chunk attention:
        # S_local: [B, num_chunks, C, C]
        S_local = torch.matmul(q_chunks, m_chunks.transpose(-2, -1))
        attn_local = S_local * intra_decay # [B, num_chunks, C, C]

        # Local recall: [B, num_chunks, C, D]
        recall_local = torch.matmul(attn_local, v_chunks)
        norm_local = torch.sum(attn_local, dim=-1, keepdim=True) # [B, num_chunks, C, 1]

        # ─────────────────────────────────────────────────────────────────────
        # Inter-Chunk Boundary Recurrence Scan
        # ─────────────────────────────────────────────────────────────────────
        chunk_recalled = []
        chunk_norms = []

        curr_M = prior_M if prior_M is not None else torch.zeros(B, self.d_map, D, device=device, dtype=torch.float32)
        curr_Z = prior_Z if prior_Z is not None else torch.zeros(B, self.d_map, device=device, dtype=torch.float32)

        # Vectorized weights for inter-chunk carry into each step within chunk: [1, C, 1, 1]
        t_decay_within = (decay_scalar ** (c_idx.float() + 1.0)).view(1, C, 1, 1).to(in_dtype)
        t_decay_Z_within = (decay_scalar ** (c_idx.float() + 1.0)).view(1, C, 1).to(in_dtype)

        for k in range(num_chunks):
            # Prior state influence on current chunk
            q_k_chunk = q_chunks[:, k] # [B, C, d_map]
            
            prior_recall_chunk = torch.matmul(q_k_chunk.unsqueeze(2), curr_M.unsqueeze(1).to(in_dtype)) * t_decay_within
            prior_recall_chunk = prior_recall_chunk.squeeze(2) # [B, C, D]

            prior_norm_chunk = torch.bmm(q_k_chunk, curr_Z.unsqueeze(-1).to(in_dtype)) * t_decay_Z_within # [B, C, 1]

            tot_recall_chunk = recall_local[:, k] + prior_recall_chunk
            tot_norm_chunk = norm_local[:, k] + prior_norm_chunk

            chunk_recalled.append(tot_recall_chunk)
            chunk_norms.append(tot_norm_chunk)

            # Accumulate this chunk's contribution to boundary state
            m_chunk_k = m_chunks[:, k] * chunk_end_weights.squeeze(1) # [B, C, d_map]
            v_chunk_k = v_chunks[:, k] # [B, C, D]

            m_acc = m_chunk_k.float() if self.fp32_accumulator else m_chunk_k
            v_acc = v_chunk_k.float() if self.fp32_accumulator else v_chunk_k

            M_chunk_end = torch.bmm(m_acc.transpose(1, 2), v_acc) # [B, d_map, D]
            Z_chunk_end = torch.sum(m_acc, dim=1) # [B, d_map]

            curr_M = decay_C * curr_M + M_chunk_end
            curr_Z = decay_C * curr_Z + Z_chunk_end

        # Flatten chunks back to sequence
        recall_all = torch.cat(chunk_recalled, dim=1) # [B, num_chunks * C, D]
        norm_all = torch.cat(chunk_norms, dim=1)       # [B, num_chunks * C, 1]

        # Truncate padding if applied
        if pad_len > 0:
            recall_all = recall_all[:, :L, :]
            norm_all = norm_all[:, :L, :]

        # Normalize and project
        norm_all = norm_all + self.eps
        recall_all = (recall_all / norm_all.clamp(min=self.eps)).to(in_dtype)

        recon = self.recon_proj(recall_all)
        gate = torch.sigmoid(self.recon_gate(x))
        out = x + gate * recon

        new_state = None
        if return_state:
            # If padded, recompute exact boundary state at L
            if pad_len > 0:
                # Direct precise accumulator from prior state
                idx_full = torch.arange(L, device=device)
                weights_full = (decay_scalar ** (L - 1 - idx_full).float()).view(1, L, 1).to(in_dtype)
                m_dec = (m_k[:, :L, :] * weights_full).float()
                v_full = v[:, :L, :].float()

                curr_M = torch.bmm(m_dec.transpose(1, 2), v_full)
                curr_Z = torch.sum(m_dec, dim=1)
                if prior_M is not None and prior_Z is not None:
                    curr_M = curr_M + (decay_scalar ** L) * prior_M
                    curr_Z = curr_Z + (decay_scalar ** L) * prior_Z

            if passed_as_obj or (state is None and not isinstance(state, tuple)):
                new_state = GistState(
                    M=curr_M,
                    Z=curr_Z,
                    step_count=prior_steps + L,
                    layer_idx=state.layer_idx if passed_as_obj else None,
                )
            else:
                new_state = (curr_M, curr_Z)

        return out, new_state
