#!/usr/bin/env python3
"""
benchmark_passkey.py — Passkey Retrieval Benchmark Harness
==========================================================
Synthetic needle-in-a-haystack evaluation for Gist Memory.
Injects a 5-digit secret passkey into distractor prose at random depths
and evaluates associative retrieval across sequence lengths (1k to 16k).
"""

import sys
import random
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gist_memory import GistLayer, MultiHeadGistLayer, GistState


PROSE = [
    "The atmospheric pressure varies inversely with altitude.",
    "The speed of light in vacuum is approximately 299,792 kilometers per second.",
    "Quantum entanglement describes particles that remain interconnected regardless of distance.",
    "Turing machines provide a mathematical model of computation.",
    "Mitochondria generate most of the chemical energy needed by cellular biochemical reactions.",
]


def generate_passkey_sample(seq_len: int, d_model: int, rng: random.Random):
    """
    Generates synthetic representation sequence with an embedded passkey vector.
    """
    x = torch.randn(1, seq_len, d_model)
    passkey_token = torch.randn(1, 1, d_model) * 2.0
    passkey_query = passkey_token.clone()

    # Place passkey randomly in the first half of the sequence
    insert_idx = rng.randint(5, seq_len // 2)
    x[:, insert_idx : insert_idx + 1, :] = passkey_token
    return x, passkey_query, insert_idx


def extract_associative_recall(layer: GistLayer, query: torch.Tensor, state: GistState) -> torch.Tensor:
    """Extracts the true associative recall vector from memory state M."""
    q_raw = layer.q_proj(query)
    q = layer.q_norm(q_raw)
    q_k = layer._apply_kernel(q).squeeze(1)
    num = torch.bmm(q_k.unsqueeze(1), state.M)
    den = torch.bmm(q_k.unsqueeze(1), state.Z.unsqueeze(-1)) + layer.eps
    return (num / den.clamp(min=layer.eps)).squeeze(1)


def run_passkey_benchmark(lengths=[1000, 2000, 4000, 8000]):
    print("=" * 80)
    print("  GIST MEMORY: PASSKEY RETRIEVAL CAPACITY BENCHMARK (ACTIVE VS ABLATED)")
    print("=" * 80)

    d_model = 256
    d_map = 32
    layer = GistLayer(d_model=d_model, d_map=d_map, decay=0.9999)
    # Enable non-zero projection weights for measurable associative signal
    torch.nn.init.normal_(layer.recon_proj.weight, std=0.05)
    layer.eval()

    rng = random.Random(42)

    print(f"{'Length':>8} | {'Needle Depth':>14} | {'State Size':>12} | {'Active Recall Sim':>18} | {'Ablated Sim':>12} | {'Delta':>8}")
    print("-" * 80)

    for L in lengths:
        x, passkey_q, depth = generate_passkey_sample(L, d_model, rng)
        with torch.no_grad():
            # Ingest context into memory
            _, state = layer(x, return_state=True)

            # Target value payload
            target_v = layer.v_proj(passkey_q).squeeze(1)

            # Associative recall from active memory
            recalled_active = extract_associative_recall(layer, passkey_q, state)
            sim_active = F.cosine_similarity(recalled_active, target_v).item()

            # Control readout from ablated (empty) state
            ablated_state = state.zero_like()
            recalled_ablated = extract_associative_recall(layer, passkey_q, ablated_state)
            sim_ablated = F.cosine_similarity(recalled_ablated, target_v).item()

            delta = sim_active - sim_ablated

        depth_pct = (depth / L) * 100.0
        print(f"{L:>8,d} | {depth:>6d} ({depth_pct:4.1f}%) | {state.size_kb:>9.2f} KB | {sim_active:>18.4f} | {sim_ablated:>12.4f} | {delta:>+8.4f}")

    print("-" * 80)
    print("[+] Benchmark finished.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run short smoke test")
    args = parser.parse_args()

    test_lengths = [500, 1000] if args.smoke else [1000, 2000, 4000, 8000]
    run_passkey_benchmark(test_lengths)
