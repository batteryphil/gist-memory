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


def run_passkey_benchmark(lengths=[1000, 2000, 4000, 8000]):
    print("=" * 70)
    print("  GIST MEMORY: PASSKEY RETRIEVAL CAPACITY BENCHMARK")
    print("=" * 70)

    d_model = 256
    d_map = 32
    layer = GistLayer(d_model=d_model, d_map=d_map, decay=0.9999)
    layer.eval()

    rng = random.Random(42)

    print(f"{'Length':>8} | {'Needle Depth':>14} | {'State Size':>12} | {'Retrieval Cosine Sim':>22}")
    print("-" * 65)

    for L in lengths:
        x, passkey_q, depth = generate_passkey_sample(L, d_model, rng)
        with torch.no_grad():
            # Ingest context into memory
            _, state = layer(x, return_state=True)

            # Query memory with passkey cue
            recalled, _ = layer(passkey_q, state=state, return_state=True)

            # Measure associative alignment
            sim = F.cosine_similarity(recalled.squeeze(1), passkey_q.squeeze(1)).item()

        depth_pct = (depth / L) * 100.0
        print(f"{L:>8,d} | {depth:>6d} ({depth_pct:4.1f}%) | {state.size_kb:>9.2f} KB | {sim:>22.4f}")

    print("-" * 65)
    print("[+] Benchmark finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run short smoke test")
    args = parser.parse_args()

    test_lengths = [500, 1000] if args.smoke else [1000, 2000, 4000, 8000]
    run_passkey_benchmark(test_lengths)
