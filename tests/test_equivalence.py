"""
test_equivalence.py — Numerical Equivalence Test Suite
======================================================
Proves that:
1. Parallel Training Mode == Autoregressive Streaming Step Mode (within float tolerance)
2. Chunked SSD Mode == Parallel Full Sequence Mode (within float tolerance)
3. Chunked State Passing across chunk boundaries preserves exact (M, Z) state
"""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gist_memory import GistLayer, ChunkedGistLayer, GistState


def test_parallel_vs_streaming_equivalence(d_model=256, d_map=32, kernel="square"):
    torch.manual_seed(42)
    seq_len = 24
    layer = GistLayer(
        d_model=d_model,
        d_map=d_map,
        decay=0.995,
        kernel=kernel,
        fp32_accumulator=True,
    )
    layer.eval()

    x = torch.randn(2, seq_len, d_model)

    # 1. Parallel forward pass
    with torch.no_grad():
        out_parallel, state_parallel = layer(x, return_state=True)

    # 2. Sequential autoregressive streaming steps
    out_streaming_steps = []
    state_stream = None
    with torch.no_grad():
        for t in range(seq_len):
            x_t = x[:, t : t + 1, :]
            out_t, state_stream = layer(x_t, state=state_stream, return_state=True)
            out_streaming_steps.append(out_t)

    out_streaming = torch.cat(out_streaming_steps, dim=1)

    # 3. Assert mathematical equality
    diff_out = (out_parallel - out_streaming).abs().max().item()
    diff_M = (state_parallel.M - state_stream.M).abs().max().item()
    diff_Z = (state_parallel.Z - state_stream.Z).abs().max().item()

    print(f"  [{kernel:6s}] d_model={d_model:3d}, d_map={d_map:2d}: OutDiff={diff_out:.2e}, MDiff={diff_M:.2e}, ZDiff={diff_Z:.2e}")
    assert diff_out < 1e-4, f"Output divergence exceeds tolerance: {diff_out}"
    assert diff_M < 1e-4, f"State M divergence exceeds tolerance: {diff_M}"
    assert diff_Z < 1e-4, f"State Z divergence exceeds tolerance: {diff_Z}"


def test_chunked_ssd_vs_full_sequence():
    torch.manual_seed(1337)
    d_model = 256
    d_map = 32
    seq_len = 160 # Multiple chunks (chunk_size=32 -> 5 chunks)
    chunk_size = 32

    base_layer = GistLayer(d_model=d_model, d_map=d_map, decay=0.999)
    chunked_layer = ChunkedGistLayer(d_model=d_model, d_map=d_map, chunk_size=chunk_size, decay=0.999)

    # Copy identical weights
    chunked_layer.load_state_dict(base_layer.state_dict())
    base_layer.eval()
    chunked_layer.eval()

    x = torch.randn(2, seq_len, d_model)

    with torch.no_grad():
        out_base, state_base = base_layer(x, return_state=True)
        out_chunked, state_chunked = chunked_layer(x, return_state=True)

    diff_out = (out_base - out_chunked).abs().max().item()
    diff_M = (state_base.M - state_chunked.M).abs().max().item()
    diff_Z = (state_base.Z - state_chunked.Z).abs().max().item()

    print(f"  [Chunked SSD vs Full] OutDiff={diff_out:.2e}, MDiff={diff_M:.2e}, ZDiff={diff_Z:.2e}")
    assert diff_out < 1e-4, f"Chunked output divergence exceeds tolerance: {diff_out}"
    assert diff_M < 1e-4, f"Chunked state M divergence exceeds tolerance: {diff_M}"
    assert diff_Z < 1e-4, f"Chunked state Z divergence exceeds tolerance: {diff_Z}"


if __name__ == "__main__":
    for k in ["square", "relu2", "elu1"]:
        test_parallel_vs_streaming_equivalence(128, 16, k)
        test_parallel_vs_streaming_equivalence(256, 32, k)
    test_chunked_ssd_vs_full_sequence()
    print("[SUCCESS] All equivalence tests passed!")
