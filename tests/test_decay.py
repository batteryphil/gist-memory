"""
test_decay.py — Verification of Multi-Scale and Data-Dependent Decays
=====================================================================
"""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gist_memory import MultiScaleDecay, DataDependentDecay


def test_multiscale_decay_initialization():
    decay = MultiScaleDecay(
        num_channels=4,
        min_half_life=8.0,
        max_half_life=8192.0,
        learnable=True,
    )
    rates = decay()
    half_lives = decay.get_half_lives()

    assert rates.shape == (4,)
    assert (rates > 0.0).all()
    assert (rates < 1.0).all()

    for i in range(len(half_lives) - 1):
        assert half_lives[i] < half_lives[i + 1]
    print(f"  [+] MultiScaleDecay half-lives: {half_lives.tolist()}")


def test_data_dependent_decay():
    B, L, D = 2, 8, 64
    num_channels = 4
    dd_decay = DataDependentDecay(d_model=D, num_channels=num_channels, base_half_life=512.0)

    x = torch.randn(B, L, D)
    decays = dd_decay(x)

    assert decays.shape == (B, L, num_channels)
    assert (decays > 0.0).all()
    assert (decays < 1.0).all()
    print("  [+] DataDependentDecay output verified in (0, 1).")


if __name__ == "__main__":
    test_multiscale_decay_initialization()
    test_data_dependent_decay()
    print("[SUCCESS] All decay tests passed!")
