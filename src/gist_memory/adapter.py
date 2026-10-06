"""
adapter.py — Hugging Face adapter for Gist Memory
=================================================
Wraps selected decoder layers of a pretrained model with a GistLayer whose
output is added (through a zero-initialized, gated residual) to the layer output.

Important — what this does NOT do:
- It does NOT replace or shrink the model's attention / KV cache. The original
  layer (including its attention and KV cache) still runs in full; Gist is an
  additional recurrent memory on top. To actually save memory you would have to
  restrict the base attention (e.g. a sliding window) and train Gist to carry
  long-range information. No such training is included in this repository.

What it does provide:
1. Zero-initialized output projection, so the wrapped model's outputs match the
   base model at step 0 (check with ``GistModelAdapter.verify_zero_init``).
2. A per-layer recurrent state carried through ``past_key_values`` / GistCache.
3. ``freeze_backbone()`` for training only Gist parameters.
"""

from __future__ import annotations
import inspect
from typing import Optional, List, Dict, Union, Tuple, Any
from pathlib import Path
import torch
import torch.nn as nn

from .core import GistLayer
from .multihead import MultiHeadGistLayer
from .state import GistState


class GistCache:
    """
    State cache compatible with Hugging Face generation loops.
    Stores and propagates GistState per layer across autoregressive decoding steps.
    """
    def __init__(self, base_cache: Optional[Any] = None):
        self.base_cache = base_cache
        self.gist_states: Dict[int, Union[GistState, Tuple[torch.Tensor, torch.Tensor]]] = {}

    def get_gist_state(self, layer_idx: int) -> Optional[Union[GistState, Tuple[torch.Tensor, torch.Tensor]]]:
        return self.gist_states.get(layer_idx, None)

    def set_gist_state(self, layer_idx: int, state: Union[GistState, Tuple[torch.Tensor, torch.Tensor]]) -> None:
        self.gist_states[layer_idx] = state

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        """Reorders cached Gist states across beams for beam search."""
        if self.base_cache is not None and hasattr(self.base_cache, "reorder_cache"):
            self.base_cache.reorder_cache(beam_idx)
        for layer_idx, state in list(self.gist_states.items()):
            if isinstance(state, GistState):
                b_idx = beam_idx.to(state.M.device)
                state.M = state.M.index_select(0, b_idx)
                state.Z = state.Z.index_select(0, b_idx)
            elif isinstance(state, tuple):
                m, z = state
                b_idx = beam_idx.to(m.device)
                self.gist_states[layer_idx] = (m.index_select(0, b_idx), z.index_select(0, b_idx))

    def __getattr__(self, name: str):
        # Forward any unhandled attributes to base Hugging Face Cache (e.g. DynamicCache)
        if self.base_cache is not None:
            return getattr(self.base_cache, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class GistWrapperLayer(nn.Module):
    """
    Wraps an individual transformer/SSM layer with a Gist associative memory module.
    """
    def __init__(
        self,
        orig_layer: nn.Module,
        layer_idx: int,
        d_model: int,
        d_map: int = 32,
        multihead: bool = False,
        num_heads: int = 4,
        decay: float = 0.9995,
        eps: float = 1e-4,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.orig_layer = orig_layer
        self.layer_idx = layer_idx
        self.d_model = d_model

        if multihead:
            self.gist = MultiHeadGistLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_map=d_map,
                eps=eps,
            )
        else:
            self.gist = GistLayer(
                d_model=d_model,
                d_map=d_map,
                decay=decay,
                eps=eps,
            )

        # Enforce strict zero initialization: base model unchanged at step 0
        nn.init.zeros_(self.gist.recon_proj.weight)
        if hasattr(self.gist.recon_proj, "bias") and self.gist.recon_proj.bias is not None:
            nn.init.zeros_(self.gist.recon_proj.bias)

        self.current_state: Optional[Union[GistState, Tuple[torch.Tensor, torch.Tensor]]] = None
        self.pinned_state: Optional[Union[GistState, Tuple[torch.Tensor, torch.Tensor]]] = None
        self.bypass: bool = False  # if True, behave exactly like orig_layer

        if device is not None or dtype is not None:
            self.gist.to(device=device, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor, *args, past_key_values: Optional[Any] = None, **kwargs):
        # 1. Execute original transformer layer
        try:
            layer_outputs = self.orig_layer(hidden_states, *args, past_key_values=past_key_values, **kwargs)
        except TypeError as e:
            if "past_key_values" in str(e):
                layer_outputs = self.orig_layer(hidden_states, *args, **kwargs)
            else:
                raise e
        if isinstance(layer_outputs, tuple):
            orig_hidden = layer_outputs[0]
        else:
            orig_hidden = layer_outputs

        if self.bypass:
            return layer_outputs

        # 2. Extract or resolve Gist state from cache or pinned state
        prior_state = None
        if past_key_values is not None:
            if hasattr(past_key_values, "get_gist_state"):
                prior_state = past_key_values.get_gist_state(self.layer_idx)
            elif hasattr(past_key_values, "gist_states"):
                prior_state = past_key_values.gist_states.get(self.layer_idx, None)
            else:
                # Dynamically attach dictionary to custom or standard DynamicCache
                try:
                    past_key_values.gist_states = {}
                    prior_state = None
                except Exception:
                    pass
        if prior_state is None and self.pinned_state is not None:
            prior_state = self.pinned_state

        # Extract attention_mask if present
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is None and len(args) > 1 and isinstance(args[1], torch.Tensor):
            attention_mask = args[1]

        # 3. Memory forward pass
        gist_out, new_state = self.gist(orig_hidden, state=prior_state, return_state=True, attention_mask=attention_mask)
        self.current_state = new_state

        # 4. Save updated state back into cache
        if past_key_values is not None and new_state is not None:
            if hasattr(past_key_values, "set_gist_state"):
                past_key_values.set_gist_state(self.layer_idx, new_state)
            elif hasattr(past_key_values, "gist_states"):
                past_key_values.gist_states[self.layer_idx] = new_state

        if isinstance(layer_outputs, tuple):
            return (gist_out,) + layer_outputs[1:]
        else:
            return gist_out



class GistModelAdapter:
    """
    Orchestrates Gist injection into full Hugging Face models.
    """
    def __init__(
        self,
        model: nn.Module,
        target_layers: List[int],
        d_model: Optional[int] = None,
        d_map: int = 32,
        multihead: bool = False,
        num_heads: int = 4,
        decay: float = 0.9995,
    ):
        self.model = model
        self.target_layers = sorted(target_layers)
        self.layers_module = self._find_layers_container(model)

        if d_model is None:
            d_model = getattr(model.config, "hidden_size", None)
            if d_model is None:
                raise ValueError("Could not automatically infer hidden_size from model.config. Specify d_model.")
        self.d_model = d_model

        # Determine target device and dtype
        sample_param = next(model.parameters())
        device = sample_param.device
        dtype = sample_param.dtype

        # Patch layers
        self.wrapped_layers: Dict[int, GistWrapperLayer] = {}
        for idx in self.target_layers:
            if idx < len(self.layers_module):
                orig_layer = self.layers_module[idx]
                wrapped = GistWrapperLayer(
                    orig_layer=orig_layer,
                    layer_idx=idx,
                    d_model=d_model,
                    d_map=d_map,
                    multihead=multihead,
                    num_heads=num_heads,
                    decay=decay,
                    dtype=dtype,
                    device=device,
                )
                self.layers_module[idx] = wrapped
                self.wrapped_layers[idx] = wrapped

    def _find_layers_container(self, model: nn.Module) -> nn.ModuleList:
        """Finds the ModuleList holding trunk decoder layers."""
        candidates = [
            "model.layers",
            "transformer.h",
            "gpt_neox.layers",
            "backbone.layers",
            "layers",
        ]
        for path in candidates:
            curr = model
            parts = path.split(".")
            found = True
            for p in parts:
                if hasattr(curr, p):
                    curr = getattr(curr, p)
                else:
                    found = False
                    break
            if found and isinstance(curr, (nn.ModuleList, list)):
                return curr
        raise AttributeError(f"Could not locate decoder layers list in model: {type(model).__name__}")

    def freeze_backbone(self) -> Dict[str, int]:
        """Freezes all trunk weights; unfreezes only Gist associative memory parameters."""
        for p in self.model.parameters():
            p.requires_grad = False

        gist_params = 0
        for wrapped in self.wrapped_layers.values():
            for p in wrapped.gist.parameters():
                p.requires_grad = True
                gist_params += p.numel()

        total_params = sum(p.numel() for p in self.model.parameters())
        return {
            "trainable_gist_params": gist_params,
            "total_params": total_params,
            "trainable_pct": (gist_params / total_params) * 100.0 if total_params > 0 else 0.0,
        }

    def get_gist_parameters(self) -> List[nn.Parameter]:
        """Returns list of all trainable Gist parameters."""
        params = []
        for wrapped in self.wrapped_layers.values():
            params.extend([p for p in wrapped.gist.parameters() if p.requires_grad])
        return params

    def save_gist_weights(self, path: Union[str, Path]) -> None:
        """Saves only Gist weights (small file, typically < 10 MB)."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        state_dict = {
            f"gist_layer_{idx}": wrapped.gist.state_dict()
            for idx, wrapped in self.wrapped_layers.items()
        }
        torch.save(state_dict, target_path)

    def load_gist_weights(self, path: Union[str, Path]) -> None:
        """Loads saved Gist weights."""
        state_dict = torch.load(path, map_location="cpu")
        for idx, wrapped in self.wrapped_layers.items():
            key = f"gist_layer_{idx}"
            if key in state_dict:
                wrapped.gist.load_state_dict(state_dict[key])

    def verify_zero_init(self, dummy_input_ids: torch.Tensor) -> float:
        """
        Measures the perturbation Gist injection causes on the base model.

        Runs the model twice — once with every Gist wrapper bypassed (pure base
        model) and once with Gist active — and returns the peak absolute
        difference between the two logit tensors. At initialization this should
        be 0.0; after training it measures how much Gist changes outputs.
        Note: Gist states are cleared before and after the check.
        """
        was_training = self.model.training
        self.model.eval()

        def _logits(out):
            return out.logits if hasattr(out, "logits") else out[0]

        try:
            with torch.no_grad():
                for w in self.wrapped_layers.values():
                    w.bypass = True
                self.clear_states()
                base_logits = _logits(self.model(dummy_input_ids, use_cache=False))

                for w in self.wrapped_layers.values():
                    w.bypass = False
                self.clear_states()
                gist_logits = _logits(self.model(dummy_input_ids, use_cache=False))
        finally:
            for w in self.wrapped_layers.values():
                w.bypass = False
            self.clear_states()
            if was_training:
                self.model.train()

        return (gist_logits.float() - base_logits.float()).abs().max().item()

    def get_states(self) -> Dict[int, Any]:
        """Returns the current GistState dictionary for all wrapped layers."""
        return {idx: w.current_state for idx, w in self.wrapped_layers.items() if w.current_state is not None}

    def set_states(self, states: Dict[int, Any]) -> None:
        """Injects / pins GistStates across wrapped layers for zero-prompt memory retrieval."""
        for idx, state in states.items():
            if idx in self.wrapped_layers:
                self.wrapped_layers[idx].pinned_state = state
                self.wrapped_layers[idx].current_state = state

    def clear_states(self) -> None:
        """Clears all pinned and current GistStates across all wrapped layers."""
        for w in self.wrapped_layers.values():
            w.pinned_state = None
            w.current_state = None

