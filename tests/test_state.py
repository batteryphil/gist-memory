"""
test_state.py — Verification of GistState Representation & Diagnostics
======================================================================
"""

import sys
import tempfile
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gist_memory import GistState


def test_gist_state_diagnostics():
    torch.manual_seed(42)
    B, d_map, D = 2, 32, 256
    M = torch.randn(B, d_map, D)
    Z = torch.rand(B, d_map) + 0.1

    state = GistState(M=M, Z=Z, step_count=100, layer_idx=7)

    assert state.batch_size == 2
    assert state.d_map == 32
    assert state.is_multihead is False
    assert state.size_kb > 0

    energy = state.state_energy()
    assert energy["M_frob_norm"] > 0
    assert energy["Z_l2_norm"] > 0
    assert energy["step_count"] == 100

    diag = state.effective_capacity()
    assert "effective_rank" in diag
    assert "condition_number" in diag
    assert diag["effective_rank"] > 0
    print(f"  [+] State Diagnostics: Effective Rank={diag['effective_rank']:.2f}, Footprint={state.size_kb:.2f} KB")


def test_gist_state_serialization():
    M = torch.randn(1, 32, 128)
    Z = torch.ones(1, 32)
    state = GistState(M=M, Z=Z, step_count=50, layer_idx=3, metadata={"author": "batteryphil"})

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_snapshot.pt"
        state.save(save_path)
        assert save_path.exists()

        loaded_state = GistState.load(save_path)
        assert loaded_state.step_count == 50
        assert loaded_state.layer_idx == 3
        assert loaded_state.metadata.get("author") == "batteryphil"
        assert torch.allclose(state.M, loaded_state.M)
        assert torch.allclose(state.Z, loaded_state.Z)
        print("  [+] State Serialization: Save & Load roundtrip successful.")


if __name__ == "__main__":
    test_gist_state_diagnostics()
    test_gist_state_serialization()
    print("[SUCCESS] All state tests passed!")
