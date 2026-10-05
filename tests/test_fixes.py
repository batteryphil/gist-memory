"""
test_fixes.py — Regression & Verification Suite for Audit Bugfixes
===================================================================
Verifies:
1. DataDependentDecay exact numerical equivalence (Parallel vs Streaming).
2. attention_mask left-padding invariance across GistLayer, MultiHeadGistLayer, and ChunkedGistLayer.
3. GistCache.reorder_cache beam search state alignment.
4. GistState.__add__ multi-source metadata chain retention.
"""

import math
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gist_memory import (
    GistLayer,
    MultiHeadGistLayer,
    ChunkedGistLayer,
    GistState,
    DataDependentDecay,
    GistCache,
)


def test_data_dependent_decay_exact_equivalence():
    """Verifies that DataDependentDecay produces identical outputs between parallel prefill and sequential streaming."""
    B, L, D = 2, 8, 64
    d_map = 16
    torch.manual_seed(42)

    decay_mod = DataDependentDecay(d_model=D, num_channels=1, base_half_life=64.0)
    layer = GistLayer(d_model=D, d_map=d_map, decay=decay_mod)
    torch.nn.init.normal_(layer.recon_proj.weight, std=0.1)
    layer.eval()

    x = torch.randn(B, L, D)

    # 1. Parallel pass
    with torch.no_grad():
        out_par, state_par = layer(x, return_state=True)

    # 2. Sequential streaming pass
    state_seq = None
    outs_seq = []
    with torch.no_grad():
        for t in range(L):
            token = x[:, t : t + 1, :]
            out_t, state_seq = layer(token, state=state_seq, return_state=True)
            outs_seq.append(out_t)
    out_seq = torch.cat(outs_seq, dim=1)

    diff_out = (out_par - out_seq).abs().max().item()
    diff_M = (state_par.M - state_seq.M).abs().max().item()
    diff_Z = (state_par.Z - state_seq.Z).abs().max().item()

    assert diff_out < 1e-5, f"DataDependentDecay output diverged: {diff_out}"
    assert diff_M < 1e-4, f"DataDependentDecay M state diverged: {diff_M}"
    assert diff_Z < 1e-4, f"DataDependentDecay Z state diverged: {diff_Z}"
    print(f"  [+] DataDependentDecay exact equivalence verified (Diff: {diff_out:.2e}).")


def test_attention_mask_left_padding_invariance():
    """Verifies that left-padding with attention_mask does not corrupt prompt representations or memory state."""
    torch.manual_seed(42)
    d_model = 64
    d_map = 16
    clean_len = 6
    pad_len = 4

    layers = [
        ("GistLayer", GistLayer(d_model=d_model, d_map=d_map)),
        ("MultiHeadGistLayer", MultiHeadGistLayer(d_model=d_model, num_heads=4, d_map=d_map // 4)),
        ("ChunkedGistLayer", ChunkedGistLayer(d_model=d_model, d_map=d_map, chunk_size=4)),
    ]

    for name, layer in layers:
        torch.nn.init.normal_(layer.recon_proj.weight, std=0.1)
        layer.eval()

        clean_seq = torch.randn(1, clean_len, d_model)
        with torch.no_grad():
            out_clean, state_clean = layer(clean_seq, return_state=True)

        # Pad with realistic non-zero vectors on the left
        pad_emb = torch.randn(1, pad_len, d_model)
        padded_seq = torch.cat([pad_emb, clean_seq], dim=1)
        mask = torch.cat([torch.zeros(1, pad_len), torch.ones(1, clean_len)], dim=1)

        with torch.no_grad():
            out_padded, state_padded = layer(padded_seq, attention_mask=mask, return_state=True)

        out_unpadded_slice = out_padded[:, pad_len:, :]
        diff_out = (out_clean - out_unpadded_slice).abs().max().item()
        diff_M = (state_clean.M - state_padded.M).abs().max().item()
        diff_Z = (state_clean.Z - state_padded.Z).abs().max().item()

        assert diff_out < 1e-5, f"{name}: Left-padding output diverged: {diff_out}"
        assert diff_M < 1e-5, f"{name}: Left-padding M state diverged: {diff_M}"
        assert diff_Z < 1e-5, f"{name}: Left-padding Z state diverged: {diff_Z}"
        print(f"  [+] {name}: Left-padding invariance verified (Diff: {diff_out:.2e}).")


def test_gist_cache_reorder_beams():
    """Verifies that GistCache correctly reorders batch indices when beam search selects new beam candidates."""
    cache = GistCache()
    B = 4
    d_map = 8
    d_model = 32

    M_init = torch.arange(B).view(B, 1, 1).repeat(1, d_map, d_model).float()
    Z_init = torch.arange(B).view(B, 1).repeat(1, d_map).float()

    state = GistState(M=M_init.clone(), Z=Z_init.clone())
    cache.set_gist_state(0, state)

    beam_idx = torch.tensor([3, 1, 0, 2])
    cache.reorder_cache(beam_idx)

    reordered_state = cache.get_gist_state(0)
    for new_b, old_b in enumerate(beam_idx.tolist()):
        assert (reordered_state.M[new_b] == old_b).all()
        assert (reordered_state.Z[new_b] == old_b).all()

    print("  [+] GistCache.reorder_cache beam index selection verified.")


def test_state_multi_source_metadata_fusion():
    """Verifies that chained additions (s1 + s2 + s3) retain all source agent identifiers."""
    s1 = GistState(M=torch.zeros(1, 4, 8), Z=torch.zeros(1, 4), metadata={"agent_id": "scout_alpha"})
    s2 = GistState(M=torch.zeros(1, 4, 8), Z=torch.zeros(1, 4), metadata={"agent_id": "auditor_beta"})
    s3 = GistState(M=torch.zeros(1, 4, 8), Z=torch.zeros(1, 4), metadata={"agent_id": "hunter_gamma"})

    fused = s1 + s2 + s3
    sources = fused.metadata.get("sources", [])
    assert sources == ["scout_alpha", "auditor_beta", "hunter_gamma"], f"Sources metadata corrupted: {sources}"
    print(f"  [+] Chained state fusion metadata verified: {sources}")


if __name__ == "__main__":
    test_data_dependent_decay_exact_equivalence()
    test_attention_mask_left_padding_invariance()
    test_gist_cache_reorder_beams()
    test_state_multi_source_metadata_fusion()
    print("[SUCCESS] All bugfix regression tests passed!")
