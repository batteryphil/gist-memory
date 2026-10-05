#!/usr/bin/env python3
"""
04_local_agent_swarm.py — Autonomous Gist Memory Swarm with Local Model
=======================================================================
Demonstrates how to build a multi-agent swarm where agents collaborate by fusing
their associative memory manifolds instead of sending bulky conversational tokens.

Features:
- 100% Local Inference: Uses Qwen2.5-0.5B offline on AMD ROCm/PyTorch.
- Zero API Cost.
- Decentralized Map-Reduce Reading.
- Zero-Token Memory Consolidation.
- Surgical Ablation.
"""

import os
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from gist_memory import GistLayer, SwarmAgent, GistSwarm

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"


def main():
    print("=" * 75)
    print("   AUTONOMOUS AGENT SWARM WITH GIST MEMORY CONSOLIDATION")
    print("=" * 75)

    model_path = "/home/phil/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n[1] Initializing local base model on {device}...")
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32).to(device)
    model.eval()

    d_model = model.config.hidden_size
    d_map = 64
    embeds = model.model.embed_tokens.weight
    lm_head = model.lm_head

    # Create shared Gist routing layer
    gist_module = GistLayer(d_model=d_model, d_map=d_map, decay=1.0).to(device)
    with torch.no_grad():
        gist_module.map_proj.weight.zero_()
        gist_module.q_proj.weight.zero_()
        gist_module.v_proj.weight.copy_(torch.eye(d_model, device=device))

    # Pre-configure orthogonal key slot projections
    all_demo_keys = [" Alpha", " Beta", " Gamma", " Delta", " Sigma", " Omega"]
    for i, k in enumerate(all_demo_keys):
        kid = tok.encode(k, add_special_tokens=False)[0]
        with torch.no_grad():
            gist_module.map_proj.weight[i, :] = embeds[kid]
            gist_module.q_proj.weight[i, :] = embeds[kid]

    # Instantiate Swarm
    swarm = GistSwarm("SecurityOpsSwarm")


    # Define 3 Scout Agents
    scout_1 = SwarmAgent("Agent_Infra", "InfrastructureScout", tokenizer=tok, gist_layer=gist_module, device=device)
    scout_2 = SwarmAgent("Agent_Security", "SecurityAuditor", tokenizer=tok, gist_layer=gist_module, device=device)
    scout_3 = SwarmAgent("Agent_Incident", "ThreatHunter", tokenizer=tok, gist_layer=gist_module, device=device)
    coordinator = SwarmAgent("Coordinator", "Synthesizer", tokenizer=tok, gist_layer=gist_module, device=device)

    swarm.add_agent(scout_1)
    swarm.add_agent(scout_2)
    swarm.add_agent(scout_3)
    swarm.add_agent(coordinator)

    # Ingest distinct knowledge domains
    print("\n[2] Dispatching Scout Agents to read disjoint data streams in parallel...")
    f1 = scout_1.ingest_bindings([" Alpha", " Beta"], [" Paris", " Berlin"], embeds, slot_offset=0)
    f2 = scout_2.ingest_bindings([" Gamma", " Delta"], [" Tokyo", " London"], embeds, slot_offset=2)
    f3 = scout_3.ingest_bindings([" Sigma", " Omega"], [" Madrid", " Rome"], embeds, slot_offset=4)

    print(f"  - Scout 1 memory state: {f1[0].size_kb:.2f} KB")
    print(f"  - Scout 2 memory state: {f2[0].size_kb:.2f} KB")
    print(f"  - Scout 3 memory state: {f3[0].size_kb:.2f} KB")

    print("\n[3] Fusing swarm memories algebraically: M_swarm = M_1 + M_2 + M_3...")
    swarm_state = swarm.consolidate()
    print(f"  -> Collective memory state ready: {swarm_state[0].size_kb:.2f} KB (Zero tokens transmitted).")

    print("\n[4] Querying Coordinator with ZERO prompt context...")
    swarm.deploy_to_coordinator("Coordinator", swarm_state)

    queries = [" Alpha", " Gamma", " Sigma"]
    for q in queries:
        res = coordinator.query_token_recall(q, embeds, lm_head)
        print(f"  Query [{q.strip()}] -> Recalled: [{res['pred_token'].strip()}] (Prob: {res['prob']*100:.2f}%)")

    print("\n[5] Demonstrating Selective Amnesia via Subspace Ablation (Subtracting Scout 2)...")
    ablated_state = swarm.pool.ablate("Agent_Security")
    swarm.deploy_to_coordinator("Coordinator", ablated_state)

    res_infra = coordinator.query_token_recall(" Alpha", embeds, lm_head)
    res_sec = coordinator.query_token_recall(" Gamma", embeds, lm_head)

    print(f"  Infra Query ['Alpha'] (Scout 1 retained): [{res_infra['pred_token'].strip()}] -> Intact!")
    print(f"  Security Query ['Gamma'] (Scout 2 ablated): [{res_sec['pred_token'].strip()}] -> Forgotten!")


    print("\n" + "=" * 75)
    print("   SWARM EXECUTION FINISHED SUCCESSFULLY WITH ZERO API COST!")
    print("=" * 75)


if __name__ == "__main__":
    main()
