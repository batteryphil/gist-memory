#!/usr/bin/env python3
"""
01_basic_usage.py — Standalone Gist Memory Quickstart
=====================================================
Demonstrates:
1. Instantiating a GistLayer with O(1) state footprint.
2. Parallel prefill over a prompt sequence.
3. Continuous autoregressive token-by-token streaming decode.
4. Inspecting manifold capacity via SVD.
"""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gist_memory import GistLayer, MultiHeadGistLayer, GistState


def main():
    print("=" * 65)
    print("  GIST MEMORY: BASIC STANDALONE USAGE")
    print("=" * 65)

    d_model = 512  # Hidden representation dimension
    d_map = 32     # Compressed topological blueprint dimension
    seq_len = 500  # Initial prompt length

    # 1. Instantiate Gist Layer
    gist = GistLayer(
        d_model=d_model,
        d_map=d_map,
        decay=0.9995,
        kernel="square",
    )
    gist.eval()

    print(f"[*] Layer configured:")
    print(f"    - Hidden dimension (D): {d_model}")
    print(f"    - Blueprint dimension (d_m): {d_map}")
    print(f"    - State size per batch item: {gist.state_bytes / 1024:.2f} KB (Fixed O(1))")

    # 2. Parallel Prefill over sequence
    prompt_tokens = torch.randn(1, seq_len, d_model)
    print(f"\n[*] Processing prefill prompt of {seq_len} tokens in parallel...")
    with torch.no_grad():
        out_prompt, state = gist(prompt_tokens, return_state=True)

    print(f"    -> Output shape: {out_prompt.shape}")
    print(f"    -> Memory manifold M shape: {state.M.shape}")
    print(f"    -> Normalizer Z shape: {state.Z.shape}")
    print(f"    -> Tokens processed: {state.step_count}")

    # 3. Inspect Memory Manifold Health via SVD
    diagnostics = state.effective_capacity()
    energy = state.state_energy()
    print(f"\n[*] Memory Diagnostics:")
    print(f"    - Effective Rank: {diagnostics['effective_rank']:.2f} / {diagnostics['max_rank']}")
    print(f"    - Condition Number: {diagnostics['condition_number']:.2f}")
    print(f"    - Frobenius Norm of M: {energy['M_frob_norm']:.4f}")

    # 4. Continuous Autoregressive Token-by-Token Generation
    decode_steps = 50
    print(f"\n[*] Commencing autoregressive decode ({decode_steps} steps) with O(1) state...")
    curr_state = state

    with torch.no_grad():
        for step in range(decode_steps):
            next_token = torch.randn(1, 1, d_model)
            out_step, curr_state = gist(next_token, state=curr_state, return_state=True)

    print(f"    -> Completed {decode_steps} steps.")
    print(f"    -> Total tokens in memory: {curr_state.step_count}")
    print(f"    -> Final memory state footprint: {curr_state.size_kb:.2f} KB (Zero expansion!)")
    print("\n[+] Done.")


if __name__ == "__main__":
    main()
