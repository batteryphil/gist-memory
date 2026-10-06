#!/usr/bin/env python3
"""
benchmark_extreme_limits.py — Scaling & sanity benchmarks for Gist Memory
=========================================================================
What each experiment measures (and what it does NOT):

1. State size & prefill throughput (1k -> 128k tokens)
   - Measures: wall-clock prefill of ONE untrained ChunkedGistLayer on random
     inputs, and the byte size of its fixed state, next to the byte size of the
     KV cache Qwen2.5-0.5B would hold (computed from its real config:
     24 layers, 2 KV heads, head_dim 64, bf16).
   - Does NOT measure: model quality, or how much information the state keeps.
     A KV cache is lossless; a fixed-size state is not.

2. Decode latency vs. context length
   - Measures: per-token decode time of one GistLayer after ACTUALLY ingesting
     H tokens, versus one softmax-attention layer (SDPA) attending over a KV
     cache of H tokens with Qwen2.5-0.5B's head layout.
   - Does NOT measure: end-to-end model latency.

3. Hand-wired associative lookup (toy)
   - Projection weights are written directly from the key tokens' embeddings.
     Recall therefore works by construction. Included only as a demo of the
     read/write mechanics, together with a never-stored-key control.

4. Capacity of a d_map=32 state (random keys/values)
   - Measures: top-1 retrieval accuracy (recalled vector closest to its own
     value among all stored values) as the number of stored pairs grows,
     with random (untrained) projections.

5. State addition / subtraction (toy, slot-disjoint)
   - Uses the hand-wired setup from (3) with non-overlapping slots per agent,
     so addition/subtraction is exact by construction.

Set GIST_MODEL to a local path or hub id (default: Qwen/Qwen2.5-0.5B).
Results are written to gist_limit_experiment_results.json in the repo root.
"""

import os
import sys
import gc
import json
import time
import math
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from gist_memory import (  # noqa: E402
    GistLayer,
    ChunkedGistLayer,
    GistState,
    SwarmAgent,
    GistSwarm,
)

MODEL_ID = os.environ.get("GIST_MODEL", "Qwen/Qwen2.5-0.5B")

# Qwen2.5-0.5B architecture (from its config.json)
QWEN_LAYERS = 24
QWEN_HIDDEN = 896
QWEN_Q_HEADS = 14
QWEN_KV_HEADS = 2
QWEN_HEAD_DIM = 64

RESULTS: Dict[str, Any] = {
    "metadata": {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "note": "Gist layers are UNTRAINED. Sizes/timings only; no quality metrics.",
    }
}


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_gpu_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def reset_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def kv_cache_bytes(seq_len: int, bytes_per_elem: int = 2) -> int:
    return 2 * QWEN_LAYERS * QWEN_KV_HEADS * QWEN_HEAD_DIM * seq_len * bytes_per_elem


def load_qwen(device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(device)
    model.eval()
    return tok, model


# =====================================================================
# EXPERIMENT 1: STATE SIZE & PREFILL THROUGHPUT
# =====================================================================
def run_experiment_1_horizon_scaling(device: str):
    print("\n" + "=" * 80)
    print("  EXPERIMENT 1: STATE SIZE & PREFILL THROUGHPUT (untrained layer, random input)")
    print("=" * 80)

    d_map, chunk_size = 32, 64
    lengths = [1024, 4096, 16384, 32768, 65536, 131072]
    records = []

    layer = ChunkedGistLayer(
        d_model=QWEN_HIDDEN, d_map=d_map, chunk_size=chunk_size,
        decay=0.9999, fp32_accumulator=True,
    ).to(device).eval()

    # Hypothetical: one Gist layer per transformer layer
    gist_total_bytes = QWEN_LAYERS * layer.state_bytes

    # Warmup so the first timing isn't dominated by kernel compilation
    with torch.no_grad():
        layer(torch.randn(1, 256, QWEN_HIDDEN, device=device), return_state=True)
    sync()

    print(f"[*] Gist state if one layer per transformer layer: {gist_total_bytes/1024:.1f} KB (fp32)")
    print(f"[*] KV cache figures: Qwen2.5-0.5B, {QWEN_LAYERS} layers x {QWEN_KV_HEADS} KV heads x {QWEN_HEAD_DIM} dims, bf16\n")
    print(f"{'Tokens':>8} | {'Qwen KV cache':>14} | {'Gist state':>11} | {'Size ratio':>10} | {'Prefill (1 layer)':>17} | {'Peak VRAM':>10}")
    print("-" * 85)

    for L in lengths:
        reset_gpu_memory()
        kv = kv_cache_bytes(L)
        x = torch.randn(1, L, QWEN_HIDDEN, device=device)
        sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            out, state = layer(x, return_state=True)
        sync()
        dt = time.perf_counter() - t0
        peak = get_gpu_memory_mb()

        rec = {
            "seq_len": L,
            "qwen_kv_cache_mb": kv / 2**20,
            "gist_state_kb_24_layers": gist_total_bytes / 1024,
            "size_ratio": kv / gist_total_bytes,
            "prefill_time_sec_single_layer": dt,
            "throughput_tokens_per_sec_single_layer": L / dt,
            "peak_vram_mb": peak,
        }
        records.append(rec)
        print(f"{L:>8,d} | {kv/2**20:>11.1f} MB | {gist_total_bytes/1024:>8.1f} KB | {rec['size_ratio']:>9.1f}x | "
              f"{dt*1000:>13.1f} ms | {peak:>7.0f} MB")
        del x, out, state

    RESULTS["experiment_1_state_size"] = records
    print("-" * 85)
    print("    Size ratio compares bytes only. A trained model would still need its own")
    print("    attention for local context; this layer alone is not a replacement.")


# =====================================================================
# EXPERIMENT 2: DECODE LATENCY — GIST vs SOFTMAX ATTENTION
# =====================================================================
def run_experiment_2_decode_latency(device: str):
    print("\n" + "=" * 80)
    print("  EXPERIMENT 2: PER-TOKEN DECODE LATENCY vs CONTEXT LENGTH (single layer)")
    print("=" * 80)

    d_map = 32
    horizons = [1000, 4000, 16000, 64000]
    steps = 100
    records = []

    gist = GistLayer(d_model=QWEN_HIDDEN, d_map=d_map, decay=0.9995).to(device).eval()
    prefill = ChunkedGistLayer(d_model=QWEN_HIDDEN, d_map=d_map, chunk_size=64, decay=0.9995).to(device).eval()
    prefill.load_state_dict(gist.state_dict(), strict=False)

    print(f"{'Context':>8} | {'Gist decode':>12} | {'SDPA attention decode':>22}")
    print("-" * 50)

    for H in horizons:
        reset_gpu_memory()
        with torch.no_grad():
            # Real ingestion of H tokens to build the state
            _, state = prefill(torch.randn(1, H, QWEN_HIDDEN, device=device), return_state=True)
            if not isinstance(state, GistState):
                state = GistState(M=state[0], Z=state[1], step_count=H)

            tok = torch.randn(1, 1, QWEN_HIDDEN, device=device)
            for _ in range(10):
                gist(tok, state=state, return_state=True)
            sync()
            t0 = time.perf_counter()
            s = state
            for _ in range(steps):
                _, s = gist(tok, state=s, return_state=True)
            sync()
            gist_us = (time.perf_counter() - t0) / steps * 1e6

            # Baseline: one GQA attention step over an H-token KV cache (Qwen layout)
            dt = torch.float16 if device == "cuda" else torch.float32
            K = torch.randn(1, QWEN_KV_HEADS, H, QWEN_HEAD_DIM, device=device, dtype=dt)
            V = torch.randn_like(K)
            K = K.repeat_interleave(QWEN_Q_HEADS // QWEN_KV_HEADS, dim=1)
            V = V.repeat_interleave(QWEN_Q_HEADS // QWEN_KV_HEADS, dim=1)
            q = torch.randn(1, QWEN_Q_HEADS, 1, QWEN_HEAD_DIM, device=device, dtype=dt)
            for _ in range(10):
                F.scaled_dot_product_attention(q, K, V)
            sync()
            t0 = time.perf_counter()
            for _ in range(steps):
                F.scaled_dot_product_attention(q, K, V)
            sync()
            attn_us = (time.perf_counter() - t0) / steps * 1e6

        records.append({"context": H, "gist_decode_us": gist_us, "sdpa_decode_us": attn_us})
        print(f"{H:>8,d} | {gist_us:>9.1f} µs | {attn_us:>19.1f} µs")
        del K, V

    RESULTS["experiment_2_decode_latency"] = records
    print("-" * 50)
    print("    Note: the Gist path here is unfused PyTorch; SDPA uses optimized kernels.")
    print("    Neither number includes the rest of the model (MLPs, projections, etc.).")


# =====================================================================
# EXPERIMENT 3: HAND-WIRED ASSOCIATIVE LOOKUP (TOY)
# =====================================================================
def _handwired_layer(model, device, d_map):
    d_model = model.config.hidden_size
    layer = GistLayer(d_model=d_model, d_map=d_map, decay=1.0).to(device).eval()
    with torch.no_grad():
        layer.map_proj.weight.zero_()
        layer.q_proj.weight.zero_()
        layer.v_proj.weight.copy_(torch.eye(d_model, device=device))
    return layer


def run_experiment_3_handwired_lookup(device: str, tok, model):
    print("\n" + "=" * 80)
    print("  EXPERIMENT 3: HAND-WIRED ASSOCIATIVE LOOKUP (TOY — weights built from keys)")
    print("=" * 80)

    embeds = model.model.embed_tokens.weight
    lm_head = model.lm_head
    keys = [" Apollo", " Falcon", " Orion", " Dragon", " Gemini", " Mercury", " Voyager", " Artemis", " Pioneer", " Cassini"]
    vals = [" Saturn", " Titan", " Jupiter", " Mars", " Venus", " Pluto", " Neptune", " Uranus", " Mercury", " Moon"]

    layer = _handwired_layer(model, device, d_map=64)
    agent = SwarmAgent("Toy", "demo", tokenizer=tok, gist_layer=layer, device=device)
    agent.ingest_bindings_handwired(keys=keys, values=vals, embed_matrix=embeds, slot_offset=0)

    correct = 0
    for k, v in zip(keys, vals):
        res = agent.query_token_recall(k, embeds, lm_head)
        ok = res["pred_token"].strip() == v.strip()
        correct += ok
        print(f"{k:>10} -> {res['pred_token']!r:>12} (want {v!r:>10}) {'MATCH' if ok else 'MISS'}")

    # Control: keys never stored
    controls = [" Hubble", " Kepler", " Juno"]
    print("  Control (never-stored keys):")
    for k in controls:
        res = agent.query_token_recall(k, embeds, lm_head)
        print(f"{k:>10} -> {res['pred_token']!r:>12}")

    acc = correct / len(keys) * 100
    print(f"[*] {correct}/{len(keys)} ({acc:.0f}%). The LLM's transformer layers are not used —")
    print("    only its embedding table and lm_head. This is a lookup table, not learned memory.")
    RESULTS["experiment_3_handwired_lookup_toy"] = {"facts": len(keys), "top1_acc_pct": acc}


# =====================================================================
# EXPERIMENT 4: CAPACITY OF A FIXED STATE (RANDOM KEYS / VALUES)
# =====================================================================
def run_experiment_4_capacity(device: str):
    print("\n" + "=" * 80)
    print("  EXPERIMENT 4: RETRIEVAL CAPACITY, d_map=32, random untrained projections")
    print("=" * 80)

    d_model, d_map = 256, 32
    counts = [4, 8, 16, 32, 64, 128, 256]
    layer = GistLayer(d_model=d_model, d_map=d_map, decay=1.0, use_salience_gate=False).to(device).eval()
    records = []

    print(f"{'Pairs N':>8} | {'Top-1 retrieval':>15} | {'Mean cos(recall, target)':>24} | {'Eff. rank':>9}")
    print("-" * 66)
    for N in counts:
        torch.manual_seed(N)
        keys = torch.randn(N, d_model, device=device)
        vals = F.normalize(torch.randn(N, d_model, device=device), dim=-1)
        with torch.no_grad():
            k = layer._apply_kernel(layer.map_norm(layer.map_proj(keys)))   # [N, d_map]
            q = layer._apply_kernel(layer.q_norm(layer.q_proj(keys)))       # [N, d_map]
            M = k.T @ vals                                                   # [d_map, D]
            Z = k.sum(0)                                                     # [d_map]
            recall = (q @ M) / (q @ Z + layer.eps).unsqueeze(-1)             # [N, D]
            sims = F.normalize(recall, dim=-1) @ vals.T                      # [N, N]
            top1 = (sims.argmax(-1) == torch.arange(N, device=device)).float().mean().item()
            diag = sims.diag().mean().item()
            eff = GistState(M=M.unsqueeze(0), Z=Z.unsqueeze(0)).effective_capacity()["effective_rank"]
        records.append({"pairs": N, "top1_retrieval": top1, "mean_target_cos": diag, "effective_rank": eff})
        print(f"{N:>8} | {top1*100:>14.1f}% | {diag:>24.3f} | {eff:>9.2f}")

    RESULTS["experiment_4_capacity"] = records
    print("-" * 66)
    print("    Untrained projections; a trained model could do better or worse.")


# =====================================================================
# EXPERIMENT 5: STATE ADDITION / SUBTRACTION (TOY, SLOT-DISJOINT)
# =====================================================================
def run_experiment_5_state_algebra(device: str, tok, model):
    print("\n" + "=" * 80)
    print("  EXPERIMENT 5: STATE ADDITION / SUBTRACTION (TOY — disjoint hand-wired slots)")
    print("=" * 80)

    embeds = model.model.embed_tokens.weight
    lm_head = model.lm_head
    layer = _handwired_layer(model, device, d_map=64)

    domains = {
        "A": ([" Alpha", " Beta", " Gamma"], [" Paris", " Berlin", " Tokyo"], 0),
        "B": ([" Delta", " Sigma", " Omega"], [" London", " Madrid", " Rome"], 3),
        "C": ([" Mars", " Jupiter", " Saturn"], [" Vienna", " Dublin", " Prague"], 6),
        "D": ([" Dollar", " Euro", " Yen"], [" Zurich", " Geneva", " Basel"], 9),
    }
    swarm = GistSwarm("toy")
    for aid, (ks, vs, off) in domains.items():
        a = SwarmAgent(aid, "worker", tokenizer=tok, gist_layer=layer, device=device)
        a.ingest_bindings_handwired(ks, vs, embeds, slot_offset=off)
        swarm.add_agent(a)

    reader = SwarmAgent("reader", "reader", tokenizer=tok, gist_layer=layer, device=device)

    def score(states, aid):
        reader.inject_states(states)
        ks, vs, _ = domains[aid]
        return sum(reader.query_token_recall(k, embeds, lm_head)["pred_token"].strip() == v.strip()
                   for k, v in zip(ks, vs))

    fused = swarm.consolidate()
    summed = {aid: score(fused, aid) for aid in domains}
    ablated = swarm.pool.ablate("C")
    after = {aid: score(ablated, aid) for aid in domains}

    print(f"  Recall per agent with summed state:   {summed}")
    print(f"  Recall per agent after removing 'C':  {after}")
    print("    Exact by construction: each agent wrote to its own key slot. With shared")
    print("    or learned key directions, contributions would overlap and interfere.")
    RESULTS["experiment_5_state_algebra_toy"] = {"summed": summed, "after_removing_C": after}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("   GIST MEMORY: SCALING & SANITY BENCHMARKS (untrained layers)")
    print(f"   Device: {RESULTS['metadata']['device_name']}")
    print("=" * 80)

    run_experiment_1_horizon_scaling(device)
    run_experiment_2_decode_latency(device)
    run_experiment_4_capacity(device)

    try:
        tok, model = load_qwen(device)
    except Exception as e:  # model not available offline
        print(f"\n[!] Skipping experiments 3 and 5 (could not load {MODEL_ID}: {e})")
    else:
        run_experiment_3_handwired_lookup(device, tok, model)
        run_experiment_5_state_algebra(device, tok, model)

    out = ROOT_DIR / "gist_limit_experiment_results.json"
    out.write_text(json.dumps(RESULTS, indent=2))
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
