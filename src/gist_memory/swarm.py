"""
swarm.py — Combining Gist states by addition / subtraction
==========================================================
Utilities for summing and subtracting GistState tensors produced by different
"agents" (i.e. different input streams processed by the same GistLayer weights).

What is actually true:
- The recurrence is linear in the inputs, so with no decay, the state from
  processing streams A and B separately and adding them equals the state from
  processing both (order-independent). With decay < 1 this only holds up to
  per-stream decay weighting.
- Subtracting a stream's state exactly removes its additive contribution to M/Z.

What is NOT implied:
- This is not "unlearning" of model weights and not a privacy guarantee. Readout
  is a normalized ratio, so other streams' recall changes too, and if streams
  share key directions their contributions interfere.
- ``ingest_bindings_handwired`` writes projection weights directly from token
  embeddings (a hand-built lookup table). Results from it demonstrate linear
  superposition with orthogonal-ish keys, not learned memory.
- Agents exchange tensors, not text — but a receiving model must share identical
  Gist weights, and no trained model consuming these states exists here.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union, Tuple, Any
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import GistLayer
from .adapter import GistModelAdapter, GistCache
from .state import GistState


@dataclass
class SwarmTelemetry:
    """Telemetry data capturing memory consumption and communication bandwidth."""
    agent_id: str
    role: str
    tokens_read: int
    state_kb: float
    effective_rank: float
    frob_norm: float


class SwarmAgent:
    """
    Autonomous worker agent in a Gist Memory Swarm.
    
    Can operate in two modes:
    1. Pretrained Model Mode: Patched Hugging Face model (e.g. Qwen2.5-0.5B) via GistModelAdapter.
    2. Direct Associative Memory Mode: Lightweight GistLayer for fast token-binding & routing.
    """
    def __init__(
        self,
        agent_id: str,
        role: str,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
        adapter: Optional[GistModelAdapter] = None,
        gist_layer: Optional[GistLayer] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.agent_id = agent_id
        self.role = role
        self.model = model
        self.tokenizer = tokenizer
        self.adapter = adapter
        self.gist_layer = gist_layer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokens_read: int = 0
        self.current_states: Dict[int, GistState] = {}

    def ingest_stream(self, text: str) -> Dict[int, GistState]:
        """
        Processes a raw text stream and updates internal Gist associative memory states.
        """
        if self.model is None or self.tokenizer is None or self.adapter is None:
            raise RuntimeError(f"Agent {self.agent_id} lacks model/tokenizer/adapter for text stream ingestion.")

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]
        self.tokens_read += seq_len

        with torch.no_grad():
            _ = self.model(**inputs)

        # Retrieve updated states from adapter
        extracted = self.adapter.get_states()
        for layer_idx, state in extracted.items():
            if isinstance(state, GistState):
                state.metadata["agent_id"] = self.agent_id
                state.metadata["role"] = self.role
                self.current_states[layer_idx] = state.clone()
            elif isinstance(state, tuple):
                m_t, z_t = state
                self.current_states[layer_idx] = GistState(
                    M=m_t.clone(),
                    Z=z_t.clone(),
                    step_count=seq_len,
                    layer_idx=layer_idx,
                    metadata={"agent_id": self.agent_id, "role": self.role},
                )

        return self.export_states()

    def ingest_bindings_handwired(
        self,
        keys: List[str],
        values: List[str],
        embed_matrix: torch.Tensor,
        slot_offset: int = 0,
    ) -> Dict[int, GistState]:
        """
        TOY DEMO — hand-wired lookup table, not learned memory.

        OVERWRITES rows of ``map_proj`` and ``q_proj`` with the key tokens'
        embeddings (one slot per key), then accumulates (key -> value) outer
        products. Recall afterwards works because the key projection was built
        from the answers. Useful only to illustrate linear superposition.
        """
        if self.gist_layer is None or self.tokenizer is None:
            raise RuntimeError(f"Agent {self.agent_id} lacks gist_layer or tokenizer for binding ingestion.")

        key_ids = [self.tokenizer.encode(k, add_special_tokens=False)[0] for k in keys]
        val_ids = [self.tokenizer.encode(v, add_special_tokens=False)[0] for v in values]
        num_items = len(keys)
        self.tokens_read += num_items * 2

        d_model = self.gist_layer.d_model
        d_map = self.gist_layer.d_map

        # Configure key mapping projections for designated slots
        with torch.no_grad():
            for i, (kid, vid) in enumerate(zip(key_ids, val_ids)):
                slot = (slot_offset + i) % d_map
                self.gist_layer.map_proj.weight[slot, :] = embed_matrix[kid].to(self.device)
                self.gist_layer.q_proj.weight[slot, :] = embed_matrix[kid].to(self.device)

            # Ingest pairs
            M_accum = torch.zeros(1, d_map, d_model, device=self.device)
            Z_accum = torch.zeros(1, d_map, device=self.device)

            for i, (kid, vid) in enumerate(zip(key_ids, val_ids)):
                k_emb = embed_matrix[kid].unsqueeze(0).unsqueeze(0).to(self.device)
                v_emb = embed_matrix[vid].unsqueeze(0).unsqueeze(0).to(self.device)

                m = self.gist_layer.map_norm(self.gist_layer.map_proj(k_emb))
                m_k = self.gist_layer._apply_kernel(m)
                v_proj = self.gist_layer.v_proj(v_emb)

                M_i = torch.bmm(m_k.transpose(1, 2), v_proj)
                Z_i = torch.sum(m_k, dim=1)

                M_accum += M_i
                Z_accum += Z_i

        state = GistState(
            M=M_accum,
            Z=Z_accum,
            step_count=num_items,
            layer_idx=0,
            metadata={"agent_id": self.agent_id, "role": self.role, "num_facts": num_items},
        )
        self.current_states[0] = state
        return self.export_states()

    # Backward-compatible alias (deprecated name hid that weights are hand-written)
    ingest_bindings = ingest_bindings_handwired

    def export_states(self) -> Dict[int, GistState]:
        """Exports a copy of the agent's current Gist states."""
        return {idx: s.clone() for idx, s in self.current_states.items()}

    def inject_states(self, states: Dict[int, GistState]) -> None:
        """Injects / loads external Gist states into this agent's memory."""
        self.current_states = {idx: s.clone() for idx, s in states.items()}
        if self.adapter is not None:
            self.adapter.set_states(self.current_states)

    def clear_memory(self) -> None:
        """Wipes the agent's working associative memory."""
        self.current_states.clear()
        if self.adapter is not None:
            self.adapter.clear_states()

    def query_token_recall(
        self,
        query_token: str,
        embed_matrix: torch.Tensor,
        lm_head: nn.Linear,
        layer_idx: int = 0,
    ) -> Dict[str, Any]:
        """
        Queries the agent's associative memory with a zero-prompt probe token
        and evaluates top-1 prediction via language model head.
        """
        if self.gist_layer is None or self.tokenizer is None:
            raise RuntimeError(f"Agent {self.agent_id} lacks gist_layer or tokenizer for query.")

        state = self.current_states.get(layer_idx, None)
        q_id = self.tokenizer.encode(query_token, add_special_tokens=False)[0]
        q_emb = embed_matrix[q_id].unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_raw = self.gist_layer.q_proj(q_emb)
            q = self.gist_layer.q_norm(q_raw)
            q_k = self.gist_layer._apply_kernel(q).squeeze(1)

            if state is None or state.M.abs().sum() == 0:
                # Blank state condition
                recalled = torch.zeros(1, self.gist_layer.d_model, device=self.device)
            else:
                num = torch.bmm(q_k.unsqueeze(1), state.M)
                den = torch.bmm(q_k.unsqueeze(1), state.Z.unsqueeze(-1)) + self.gist_layer.eps
                recalled = (num / den).squeeze(1)

            recalled_norm = recalled.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            target_scale = embed_matrix[q_id].norm().item()
            recalled_calibrated = recalled * (target_scale / recalled_norm)

            logits = lm_head(recalled_calibrated)
            probs = F.softmax(logits, dim=-1)
            pred_id = logits.argmax(dim=-1).item()
            pred_token = self.tokenizer.decode([pred_id])
            pred_prob = probs[0, pred_id].item()

        return {
            "query_token": query_token,
            "pred_token": pred_token,
            "pred_token_id": pred_id,
            "prob": pred_prob,
            "recalled_norm": recalled.norm().item(),
        }

    def telemetry(self) -> List[SwarmTelemetry]:
        """Returns diagnostic telemetry for all layers in this agent."""
        res = []
        for idx, s in self.current_states.items():
            cap = s.effective_capacity()
            energy = s.state_energy()
            res.append(
                SwarmTelemetry(
                    agent_id=self.agent_id,
                    role=self.role,
                    tokens_read=self.tokens_read,
                    state_kb=s.size_kb,
                    effective_rank=cap["effective_rank"],
                    frob_norm=energy["M_frob_norm"],
                )
            )
        return res


class SwarmMemoryPool:
    """
    Central / federated memory repository for a swarm.
    Maintains registered agent states and performs mathematical state fusion and ablation.
    """
    def __init__(self):
        self.registry: Dict[str, Dict[int, GistState]] = {}
        self.history: List[Dict[str, Any]] = []

    def register_agent_memory(self, agent_id: str, states: Dict[int, GistState]) -> None:
        """Registers or updates the associative memory snapshot for an agent."""
        self.registry[agent_id] = {idx: s.clone() for idx, s in states.items()}
        self.history.append({
            "action": "register",
            "agent_id": agent_id,
            "layers": list(states.keys()),
            "total_kb": sum(s.size_kb for s in states.values()),
        })

    def fuse(
        self,
        agent_ids: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[int, GistState]:
        """
        Lossless linear manifold fusion across specified (or all) registered agents:
            M_fused^(l) = sum_i(w_i * M_i^(l))
            Z_fused^(l) = sum_i(w_i * Z_i^(l))
        """
        targets = agent_ids if agent_ids is not None else list(self.registry.keys())
        if not targets:
            return {}

        weights = weights or {aid: 1.0 for aid in targets}
        fused_by_layer: Dict[int, GistState] = {}

        for aid in targets:
            agent_mem = self.registry.get(aid, {})
            w = weights.get(aid, 1.0)
            for layer_idx, state in agent_mem.items():
                weighted_state = state * w
                if layer_idx not in fused_by_layer:
                    fused_by_layer[layer_idx] = weighted_state
                else:
                    fused_by_layer[layer_idx] = fused_by_layer[layer_idx] + weighted_state

        return fused_by_layer

    def ablate(self, target_agent_id: str) -> Dict[int, GistState]:
        """
        Re-fuses all agents EXCEPT target_agent_id (equivalently M_fused - M_target).
        This removes the target's additive contribution to M/Z exactly. It is not
        weight unlearning, and recall for remaining agents can still change.
        """
        remaining = [aid for aid in self.registry.keys() if aid != target_agent_id]
        return self.fuse(agent_ids=remaining)

    def bandwidth_audit(
        self,
        total_tokens_read: int,
        num_layers: int = 24,
        num_kv_heads: int = 2,
        head_dim: int = 64,
        bytes_per_elem: int = 2,
    ) -> Dict[str, Any]:
        """
        Byte-size comparison of the registered Gist states vs. the KV cache a
        GQA transformer would hold for the same number of tokens.
        Defaults match Qwen2.5-0.5B (24 layers, 2 KV heads, head_dim 64).

        Caveat: this compares storage size only. It says nothing about how much
        information each representation preserves — a KV cache is lossless,
        a fixed-size Gist state is not.
        """
        # 1. Total Gist state transferred across registered agents
        total_gist_bytes = sum(
            sum(s.num_bytes for s in mem.values()) for mem in self.registry.values()
        )
        total_gist_kb = total_gist_bytes / 1024.0

        # 2. KV cache size = 2 (K+V) * layers * kv_heads * head_dim * tokens * bytes
        traditional_kv_bytes = 2 * num_layers * num_kv_heads * head_dim * total_tokens_read * bytes_per_elem
        traditional_kv_mb = traditional_kv_bytes / (1024.0 * 1024.0)

        # 3. Compression ratio: KV Cache vs Gist associative manifold
        kv_compression_ratio = traditional_kv_bytes / max(1, total_gist_bytes)
        kv_savings_pct = (1.0 - (total_gist_bytes / max(1, traditional_kv_bytes))) * 100.0

        return {
            "num_agents": len(self.registry),
            "tokens_read_across_swarm": total_tokens_read,
            "tokens_transmitted_to_coordinator": 0,
            "total_gist_bytes": total_gist_bytes,
            "total_gist_kb": total_gist_kb,
            "traditional_kv_bytes": traditional_kv_bytes,
            "traditional_kv_mb": traditional_kv_mb,
            "kv_compression_ratio": kv_compression_ratio,
            "kv_savings_pct": kv_savings_pct,
        }



class GistSwarm:
    """
    Top-level orchestrator for a multi-agent Gist memory swarm.
    """
    def __init__(self, name: str = "GistSwarm"):
        self.name = name
        self.agents: Dict[str, SwarmAgent] = {}
        self.pool = SwarmMemoryPool()

    def add_agent(self, agent: SwarmAgent) -> None:
        """Registers a worker agent into the swarm."""
        self.agents[agent.agent_id] = agent

    def get_agent(self, agent_id: str) -> SwarmAgent:
        return self.agents[agent_id]

    def consolidate(self, agent_ids: Optional[List[str]] = None) -> Dict[int, GistState]:
        """
        Synchronizes all worker memories into the central pool and returns the fused manifold.
        """
        targets = agent_ids if agent_ids is not None else list(self.agents.keys())
        for aid in targets:
            agent = self.agents[aid]
            self.pool.register_agent_memory(aid, agent.export_states())
        return self.pool.fuse(agent_ids=targets)

    def deploy_to_coordinator(
        self,
        coordinator_id: str,
        memory_states: Optional[Dict[int, GistState]] = None,
    ) -> None:
        """
        Deploys consolidated swarm memory to the executive coordinator agent.
        """
        coordinator = self.agents[coordinator_id]
        states_to_inject = memory_states if memory_states is not None else self.consolidate()
        coordinator.inject_states(states_to_inject)
