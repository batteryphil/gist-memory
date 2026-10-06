"""
test_hf_adapter.py — Verification of Transformer Surgery and Zero-Init Preservation
===================================================================================
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gist_memory import GistModelAdapter, GistCache, GistWrapperLayer


class MockDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.self_attn = nn.Linear(hidden_size, hidden_size)
        self.mlp = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, *args, **kwargs):
        residual = hidden_states
        hidden_states = self.self_attn(hidden_states) + residual
        hidden_states = self.mlp(hidden_states) + hidden_states
        return (hidden_states,)


class MockConfig:
    def __init__(self, hidden_size: int, num_hidden_layers: int, vocab_size: int):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size


class MockCausalLM(nn.Module):
    def __init__(self, hidden_size: int = 128, num_layers: int = 6, vocab_size: int = 256):
        super().__init__()
        self.config = MockConfig(hidden_size, num_layers, vocab_size)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.model.layers = nn.ModuleList([MockDecoderLayer(hidden_size) for _ in range(num_layers)])
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, past_key_values=None, **kwargs):
        h = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            out = layer(h, past_key_values=past_key_values)
            h = out[0] if isinstance(out, tuple) else out
        logits = self.lm_head(h)
        return logits


def test_zero_init_preservation():
    torch.manual_seed(42)
    model = MockCausalLM(hidden_size=64, num_layers=4, vocab_size=100)
    input_ids = torch.randint(0, 100, (1, 16))

    model.eval()
    with torch.no_grad():
        base_logits = model(input_ids)

    adapter = GistModelAdapter(
        model=model,
        target_layers=[1, 3],
        d_map=16,
    )

    with torch.no_grad():
        patched_logits = model(input_ids)

    diff = (base_logits - patched_logits).abs().max().item()
    print(f"  [+] Base vs Patched Logit Max Diff: {diff:.2e}")
    assert diff == 0.0, f"Zero-init guarantee violated! Non-zero perturbation: {diff}"

    freeze_stats = adapter.freeze_backbone()
    assert freeze_stats["trainable_gist_params"] > 0
    assert not model.model.embed_tokens.weight.requires_grad
    assert not model.lm_head.weight.requires_grad
    print(f"  [+] Backbone frozen. Trainable Gist parameters: {freeze_stats['trainable_gist_params']:,} ({freeze_stats['trainable_pct']:.2f}%)")


def test_gist_cache_propagation():
    torch.manual_seed(42)
    model = MockCausalLM(hidden_size=64, num_layers=4, vocab_size=100)
    adapter = GistModelAdapter(model=model, target_layers=[1, 3], d_map=16)

    cache = GistCache()
    input_ids = torch.randint(0, 100, (1, 1))

    out1 = model(input_ids, past_key_values=cache)
    assert 1 in cache.gist_states
    assert 3 in cache.gist_states

    out2 = model(input_ids, past_key_values=cache)
    assert out2 is not None
    print("  [+] GistCache successfully carries recurrent states across generation steps.")


def test_verify_zero_init_detects_perturbation():
    torch.manual_seed(0)
    model = MockCausalLM(hidden_size=64, num_layers=4, vocab_size=100)
    adapter = GistModelAdapter(model=model, target_layers=[1, 3], d_map=16)
    ids = torch.randint(0, 100, (1, 16))

    assert adapter.verify_zero_init(ids) == 0.0

    with torch.no_grad():
        adapter.wrapped_layers[1].gist.recon_proj.weight.normal_(std=0.1)
    diff = adapter.verify_zero_init(ids)
    print(f"  [+] verify_zero_init after perturbing recon_proj: {diff:.3e}")
    assert diff > 0.0, "verify_zero_init failed to detect a non-zero Gist contribution"


if __name__ == "__main__":
    test_zero_init_preservation()
    test_gist_cache_propagation()
    test_verify_zero_init_detects_perturbation()
    print("[SUCCESS] All HF adapter tests passed!")
