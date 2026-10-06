# Gist Memory: What the Benchmarks Show (and Don't)

This replaces an earlier version of this document whose headline figures were not supported by the code. Corrections:

| Earlier claim | Correction |
|---|---|
| 10.5 GB KV cache for Qwen2.5-0.5B at 128k; "3,995× compression" | The calculation used 14 (query) heads (or "8 heads × 128 dims"). Qwen2.5-0.5B has **2 KV heads × 64 dims**, so its bf16 KV cache at 128k is **~1.5 GB**. In any case, a byte ratio says nothing about information retained, and the adapter does not remove the KV cache. |
| "O(1) decode latency across 50,000 tokens" | The benchmark reused the same random state for each "horizon"; no tokens were ingested. Fixed-cost decode is true by construction of the recurrence but was not demonstrated against attention. |
| "90% zero-prompt recall on Qwen2.5" | Projection weights were copied from the key tokens' embeddings (a hand-built lookup table); the "10,000-token document" did not exist; the model's transformer layers were never run. |
| "99.36% API cost reduction" | Pure arithmetic on an assumed 64 tokens/query; no API or quality comparison. Removed. |
| "100% collective recall / 100% amnesia in 1 µs" | Each agent wrote to its own hand-assigned slot, so addition and subtraction were exact by construction. Does not generalize to learned memory and is not unlearning. |
| "~716 facts/layer for d_m=896" | d_map (key dimension) was 32/64, not 896; extrapolation invalid. |
| `verify_zero_init() == 0.0` "proves" identity | Previously returned the max of a weight just set to zero. Now compares real logits with Gist bypassed vs. active. |

## How to reproduce

```bash
pip install -e . transformers
python benchmarks/benchmark_extreme_limits.py
# optional: GIST_MODEL=/path/to/Qwen2.5-0.5B for offline use
```

Results are written to `gist_limit_experiment_results.json`. Numbers depend on hardware; none are quoted here so this document can't drift from what the script produces.

## Reading the results

1. **State size** — the Gist state is constant (24 layers × 32 × 896 × 4 B ≈ 2.6 MB); the KV cache grows linearly. This is a property of any recurrent/linear-attention memory. It is meaningful only if a trained model can actually use the state in place of distant context, which is untested.
2. **Decode latency** — Gist decode cost is flat; SDPA cost grows with context. In the reference run (AMD Radeon, ROCm), SDPA was faster up to roughly 4–8k tokens and slower beyond (e.g. ~0.8 ms vs ~7.3 ms at 64k for a single layer).
3. **Hand-wired lookup** — demonstrates read/write mechanics only. The control (never-stored keys) shows what the readout produces with no match (in the reference run, every unknown key returned the same stored value).
4. **Capacity** — with random, untrained projections, top-1 retrieval is poor even for 4 pairs (reference run: 50% at N=4, ≈0% from N=8). RMSNorm + squared-feature keys from random projections are not discriminative; usable associative recall would have to be learned. Effective rank saturates around 28 of 32.
5. **State algebra** — exact in the disjoint-slot toy. With overlapping key directions, removing one stream's state changes readout for others.

## Not yet measured
- Any language-modeling quality metric (perplexity, downstream tasks) with Gist trained.
- Long-context retrieval (passkey, RULER) with the base model's attention restricted.
- Comparison with established methods (GLA, Mamba-2 hybrids, Infini-attention, LoLCATs).
