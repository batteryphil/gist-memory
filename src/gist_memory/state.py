"""
state.py — Typed GistState Representation, Inspection, and Serialization
========================================================================
Encapsulates second-order associative memory manifold states:
  M ∈ R^(d_map x D)  (or R^(H x d_map x d_v) for multi-head)
  Z ∈ R^(d_map)      (or R^(H x d_map) for multi-head)

Provides:
1. Seamless device/dtype casting
2. Memory capacity & rank diagnostics (SVD, singular value entropy)
3. State energy and drift tracking
4. Lightweight disk serialization (< 100 KB) for cross-session persistence
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Dict, Any, Tuple
import torch


@dataclass
class GistState:
    """
    Second-order associative memory manifold state.
    
    Attributes:
        M: Associative memory tensor of shape [B, d_map, D] or [B, H, d_map, d_v].
        Z: Normalization vector of shape [B, d_map] or [B, H, d_map].
        step_count: Number of tokens accumulated into this state.
        layer_idx: Optional layer index identifier within a transformer trunk.
        metadata: Optional dictionary for tracking session, prompt ID, or task info.
    """
    M: torch.Tensor
    Z: torch.Tensor
    step_count: int = 0
    layer_idx: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def batch_size(self) -> int:
        return self.M.shape[0]

    @property
    def is_multihead(self) -> bool:
        return self.M.ndim == 4

    @property
    def d_map(self) -> int:
        return self.M.shape[2] if self.is_multihead else self.M.shape[1]

    @property
    def num_bytes(self) -> int:
        """Total memory footprint of state in bytes."""
        return (self.M.element_size() * self.M.numel()) + (self.Z.element_size() * self.Z.numel())

    @property
    def size_kb(self) -> float:
        """Total memory footprint in kilobytes."""
        return self.num_bytes / 1024.0

    def to(self, device: Optional[Union[str, torch.device]] = None, dtype: Optional[torch.dtype] = None) -> GistState:
        """Casts or moves state to specified device/dtype."""
        new_M = self.M.to(device=device, dtype=dtype) if (device is not None or dtype is not None) else self.M
        new_Z = self.Z.to(device=device, dtype=dtype) if (device is not None or dtype is not None) else self.Z
        return GistState(
            M=new_M,
            Z=new_Z,
            step_count=self.step_count,
            layer_idx=self.layer_idx,
            metadata=dict(self.metadata),
        )

    def clone(self) -> GistState:
        """Returns a deep clone of the state."""
        return GistState(
            M=self.M.clone(),
            Z=self.Z.clone(),
            step_count=self.step_count,
            layer_idx=self.layer_idx,
            metadata=dict(self.metadata),
        )

    def detach(self) -> GistState:
        """Detaches state tensors from current computation graph."""
        return GistState(
            M=self.M.detach(),
            Z=self.Z.detach(),
            step_count=self.step_count,
            layer_idx=self.layer_idx,
            metadata=dict(self.metadata),
        )

    def zero_like(self) -> GistState:
        """Creates a zeroed state with identical shape and device (useful for ablation tests)."""
        return GistState(
            M=torch.zeros_like(self.M),
            Z=torch.zeros_like(self.Z),
            step_count=0,
            layer_idx=self.layer_idx,
            metadata={"ablated": True},
        )

    def state_energy(self) -> Dict[str, float]:
        """
        Computes energy metrics for the state:
        - M_frob_norm: Frobenius norm of M
        - Z_l2_norm: L2 norm of normalizer Z
        """
        with torch.no_grad():
            m_norm = self.M.float().norm().item()
            z_norm = self.Z.float().norm().item()
        return {
            "M_frob_norm": m_norm,
            "Z_l2_norm": z_norm,
            "step_count": self.step_count,
        }

    def effective_capacity(self) -> Dict[str, Any]:
        """
        Diagnoses manifold health via Singular Value Decomposition (SVD):
        - singular_values: Top singular values of M
        - effective_rank: Roy & Vetterli continuous entropy-based numerical rank
        - condition_number: Ratio of largest to smallest singular value
        """
        with torch.no_grad():
            M_2d = self.M.float()
            if self.is_multihead:
                # Merge batch and heads for spectral diagnosis: [B * H, d_map, d_v]
                B, H, d_map, d_v = M_2d.shape
                M_2d = M_2d.reshape(B * H, d_map, d_v)
            
            # SVD across batch items
            s = torch.linalg.svdvals(M_2d) # [*, min(d_map, D)]
            mean_s = s.mean(dim=0)
            
            # Continuous numerical rank via singular value entropy:
            # p_i = s_i / sum(s)
            # H(p) = -sum(p_i ln p_i)
            # rank = exp(H(p))
            s_sum = mean_s.sum().clamp(min=1e-12)
            p = (mean_s / s_sum).clamp(min=1e-12)
            entropy = -(p * torch.log(p)).sum().item()
            eff_rank = math.exp(entropy)
            cond = (mean_s[0] / mean_s[-1].clamp(min=1e-8)).item()

        return {
            "effective_rank": eff_rank,
            "max_rank": min(self.d_map, self.M.shape[-1]),
            "condition_number": cond,
            "top_singular_values": mean_s[:8].tolist(),
        }

    def save(self, path: Union[str, Path]) -> None:
        """Saves state to a file on disk."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "M": self.M.detach().cpu(),
            "Z": self.Z.detach().cpu(),
            "step_count": self.step_count,
            "layer_idx": self.layer_idx,
            "metadata": self.metadata,
        }
        torch.save(payload, target_path)

    @classmethod
    def load(cls, path: Union[str, Path], device: Optional[Union[str, torch.device]] = None) -> GistState:
        """Loads state snapshot from disk."""
        payload = torch.load(path, map_location=device or "cpu")
        state = cls(
            M=payload["M"],
            Z=payload["Z"],
            step_count=payload.get("step_count", 0),
            layer_idx=payload.get("layer_idx", None),
            metadata=payload.get("metadata", {}),
        )
        if device is not None:
            state = state.to(device=device)
        return state
