"""
test_multihead.py — Verification of MultiHeadGistLayer
======================================================
"""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gist_memory import MultiHeadGistLayer, GistState


def test_multihead_forward_and_shapes():
    torch.manual_seed(42)
    B, L, D = 2, 32, 256
    H = 4
    d_m = 16

    layer = MultiHeadGistLayer(
        d_model=D,
        num_heads=H,
        d_map=d_m,
        learnable_decay=True,
    )
    layer.eval()

    x = torch.randn(B, L, D)
    out, state = layer(x, return_state=True)

    assert out.shape == (B, L, D)
    assert isinstance(state, GistState)
    assert state.M.shape == (B, H, d_m, D // H)
    assert state.Z.shape == (B, H, d_m)
    assert state.is_multihead is True
    print(f"  [+] Multi-head shapes verified: M {state.M.shape}, Z {state.Z.shape}")


def test_multihead_gradient_flow():
    torch.manual_seed(42)
    B, L, D = 2, 16, 128
    layer = MultiHeadGistLayer(
        d_model=D,
        num_heads=4,
        d_map=16,
        learnable_decay=True,
    )
    layer.train()

    torch.nn.init.normal_(layer.recon_proj.weight, std=0.01)

    x = torch.randn(B, L, D, requires_grad=True)
    out, state = layer(x, return_state=True)

    loss = out.sum() + state.M.sum()
    loss.backward()

    assert x.grad is not None and x.grad.norm().item() > 0
    assert layer.map_proj.weight.grad is not None and layer.map_proj.weight.grad.norm().item() > 0
    assert layer.q_proj.weight.grad is not None and layer.q_proj.weight.grad.norm().item() > 0
    assert layer.v_proj.weight.grad is not None and layer.v_proj.weight.grad.norm().item() > 0
    assert layer.decay.raw_alpha.grad is not None and layer.decay.raw_alpha.grad.norm().item() > 0
    print("  [+] Multi-head gradient backpropagation verified through all heads and decay parameters.")


def test_multihead_streaming_equivalence():
    torch.manual_seed(42)
    B, L, D = 2, 16, 128
    H = 4
    d_m = 16

    layer = MultiHeadGistLayer(
        d_model=D,
        num_heads=H,
        d_map=d_m,
        learnable_decay=False,
    )
    layer.eval()

    x = torch.randn(B, L, D)

    with torch.no_grad():
        out_par, state_par = layer(x, return_state=True)

    out_steps = []
    state_stream = None
    with torch.no_grad():
        for t in range(L):
            x_t = x[:, t : t + 1, :]
            out_t, state_stream = layer(x_t, state=state_stream, return_state=True)
            out_steps.append(out_t)

    out_stream = torch.cat(out_steps, dim=1)

    diff_out = (out_par - out_stream).abs().max().item()
    diff_M = (state_par.M - state_stream.M).abs().max().item()
    diff_Z = (state_par.Z - state_stream.Z).abs().max().item()

    print(f"  [Multi-Head Equivalence] OutDiff={diff_out:.2e}, MDiff={diff_M:.2e}, ZDiff={diff_Z:.2e}")
    assert diff_out < 1e-4, f"Multihead output diff too large: {diff_out}"
    assert diff_M < 1e-4, f"Multihead state M diff too large: {diff_M}"
    assert diff_Z < 1e-4, f"Multihead state Z diff too large: {diff_Z}"


if __name__ == "__main__":
    test_multihead_forward_and_shapes()
    test_multihead_gradient_flow()
    test_multihead_streaming_equivalence()
    print("[SUCCESS] All multihead tests passed!")
