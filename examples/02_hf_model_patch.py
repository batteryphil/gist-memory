#!/usr/bin/env python3
"""
02_hf_model_patch.py — Surgical Injection into Hugging Face Models
==================================================================
Demonstrates how to patch any pretrained Transformer or SSM trunk
(e.g., DeepSeek, Qwen2, LLaMA-3, Mistral) with Gist Memory while guaranteeing
that base model performance is 100% unaltered prior to fine-tuning.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gist_memory import GistModelAdapter, GistCache


def main():
    print("=" * 65)
    print("  GIST MEMORY: HUGGING FACE MODEL PATCHING")
    print("=" * 65)

    # For fast, offline demonstration, we construct a standard transformer trunk.
    # In production, this can be AutoModelForCausalLM.from_pretrained("deepseek-ai/...")
    print("[*] Initializing host language model trunk...")
    class SimpleTransformer(nn.Module):
        def __init__(self, vocab_size=1000, hidden_size=256, num_layers=8):
            super().__init__()
            self.config = type("Config", (), {"hidden_size": hidden_size, "vocab_size": vocab_size})()
            self.model = nn.Module()
            self.model.embed = nn.Embedding(vocab_size, hidden_size)
            self.model.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dim_feedforward=512, batch_first=True)
                for _ in range(num_layers)
            ])
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        def forward(self, input_ids, past_key_values=None, **kwargs):
            h = self.model.embed(input_ids)
            for layer in self.model.layers:
                h = layer(h)
            return type("Out", (), {"logits": self.lm_head(h)})()

    model = SimpleTransformer(vocab_size=1000, hidden_size=256, num_layers=8)
    model.eval()

    test_input = torch.randint(0, 1000, (1, 16))

    # 1. Baseline output
    with torch.no_grad():
        base_logits = model(test_input).logits

    # 2. Patch mid-trunk and deep layers with Gist Memory
    target_layers = [3, 7]
    print(f"\n[*] Grafting Gist Memory onto layers: {target_layers}...")
    adapter = GistModelAdapter(
        model=model,
        target_layers=target_layers,
        d_map=32,
    )

    # 3. Verify Strict Zero-Init Baseline Equivalence
    with torch.no_grad():
        patched_logits = model(test_input).logits

    deviation = (base_logits - patched_logits).abs().max().item()
    print(f"[*] Zero-Init Verification: Peak Logit Divergence = {deviation:.2e}")
    assert deviation == 0.0, "Zero-init guarantee failed!"
    print("    -> SUCCESS: Base model weights produce 100% bitwise identical predictions.")

    # 4. Freeze trunk for Parameter-Efficient Associative Training
    freeze_stats = adapter.freeze_backbone()
    print(f"\n[*] Trunk Parameter Isolation:")
    print(f"    - Trainable Gist Parameters: {freeze_stats['trainable_gist_params']:,}")
    print(f"    - Total Model Parameters:    {freeze_stats['total_params']:,}")
    print(f"    - Trainable Fraction:         {freeze_stats['trainable_pct']:.2f}%")

    print("\n[+] Ready for continuous long-context fine-tuning.")


if __name__ == "__main__":
    main()
