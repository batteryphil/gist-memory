<div align="center">

# 🧠 Gist Memory
### Biomimetic Constructive Associative Memory for Language Models

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-brightgreen.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)]()
[![State Size](https://img.shields.io/badge/Memory%20State-O(1)%20%7E64_KB-purple.svg)]()
[![Zero-Init](https://img.shields.io/badge/Zero--Init-100%25%20Bitwise%20Safe-success.svg)]()

**A fixed-size, second-order associative memory manifold that replaces unbounded $O(N)$ KV caches with compact, $O(1)$ topological thought blueprints.**

</div>

---

## ⚡ Overview

In modern Transformers, the standard Key-Value (KV) cache acts as purely *verbatim memory*: storing uncompressed activations for every single token across time. Over long horizons, this causes:
- **Catastrophic VRAM Blowup**: $O(N)$ memory expansion (consuming 8–16+ GB of VRAM just for token caches at 64k+ context).
- **Memory-Bandwidth Bottlenecks**: Autoregressive decode slows down dramatically as GPUs become memory-bus bound.
- **Needle Dilution**: Softmax attention spreads probability mass across thousands of irrelevant filler tokens.

**Gist Memory** introduces biomimetic constructive memory inspired by *Fuzzy Trace Theory* in cognitive psychology. Instead of memorizing surface tokens verbatim, Gist Memory distills semantic concepts and invariant facts into an $O(1)$ quadratic manifold ($\sim 64\text{ KB}$ per layer) and dynamically reconstructs the cognitive trajectory upon associative query.

---

## 🔬 Mathematical Architecture

```
Incoming Hidden Stream: x_t ∈ R^D
 │
 ├──► [Topological Blueprint]:  m_t = RMSNorm(W_m x_t) ∈ R^d_m   (d_m ≪ D, e.g. 32)
 ├──► [Associative Query]:      q_t = RMSNorm(W_q x_t) ∈ R^d_m
 ├──► [Information Payload]:    v_t = W_v x_t ∈ R^D
 └──► [Novelty/Salience Gate]:  γ_t = σ(w_s^T x_t + b_s) ∈ [0, 1]
                                 │
                                 ▼
                     Kernel Map: m_sq = (m_t ⊙ m_t) · γ_t
                                 q_sq = (q_t ⊙ q_t)
                                 │
                                 ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                 O(1) Associative Manifold Accumulation                 │
 │                                                                        │
 │   Normalizer State:  Z_t = λ Z_{t-1} + m_sq                            │
 │   Memory Matrix:     M_t = λ M_{t-1} + m_sq ⊗ v_t    (M ∈ R^(d_m x D)) │
 └────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
    [Associative Recall]: recall_t = (q_sq^T M_t) / (q_sq^T Z_t + ε)
                                 │
                                 ▼
    [Generative Reconstruct]: recon_t = W_recon · recall_t  (Zero-Init)
                                 │
                                 ▼
    [Residual Injection]:    x_out = x_t + σ(W_g x_t + b_g) ⊙ recon_t
```

### Key Properties
1. **$O(1)$ Flat Memory State**: Memory footprint is fixed for all sequence lengths ($10^2$ to $10^7$ tokens).
2. **Multi-Scale Learnable Decay**: Retention timescales $\lambda_k = \exp(-\text{softplus}(\alpha_k))$ cover geometric progressions of half-lives (8 tokens to 65,536+ tokens).
3. **Multi-Head Disentanglement**: `MultiHeadGistLayer` separates entity tracking, logical connectives, and narrative gist across $H$ independent sub-manifolds.
4. **Chunked SSD Linear Scan**: Replaces quadratic $O(L^2)$ prefill matrices with $O(L \cdot C)$ block-parallel scans for 128k context training on single GPUs.
5. **Strict Zero-Initialization**: Output projection $W_{recon}$ initializes to zero, guaranteeing that grafting Gist onto a pretrained model (e.g. DeepSeek-R1, Qwen2, LLaMA-3) preserves 100% of baseline model weights and outputs at step 0.

---

## 📊 Comparison: Softmax KV Cache vs Gist Memory

| Metric | Standard Softmax KV Cache | Gist Memory (Ours) | Advantage |
| :--- | :--- | :--- | :--- |
| **Decode State Memory** | $O(N)$ (Unbounded) | **$O(1)$ (Constant)** | **Zero VRAM expansion** |
| **24-Layer VRAM at 65k** | **$12.00\text{ GB}$** | **$6.15\text{ MB}$** | **$2,047\times$ compression** |
| **Decode Step Complexity** | $O(N)$ memory bandwidth | **$O(1)$ FLOPs & Bandwidth** | Flat latency across horizon |
| **Prefill Complexity** | $O(L^2)$ Attention | **$O(L \cdot C)$ Chunked SSD** | $1,000\times$ less activation VRAM |
| **Cross-Session Storage** | Gigabytes per conversation | **$< 100\text{ KB}$ Snapshot** | Save memory snapshots to disk |
| **Pretrained Surgery** | Destructive / non-trivial | **100% Safe Zero-Init** | Identity mapping at step 0 |

---

## 🚀 Quickstart

### Installation
```bash
git clone https://github.com/batteryphil/gist-memory.git
cd gist-memory
pip install -e .
```

### 1. Minimal Standalone Usage
```python
import torch
from gist_memory import GistLayer

# Hidden size 512, blueprint dimension 32 (State size = 64 KB)
layer = GistLayer(d_model=512, d_map=32, decay=0.9995)

# Parallel prefill over 1,000 tokens
prompt = torch.randn(1, 1000, 512)
out_prompt, state = layer(prompt, return_state=True)

# Autoregressive streaming decode in O(1) memory
token = torch.randn(1, 1, 512)
out_next, state = layer(token, state=state, return_state=True)

# Inspect manifold capacity
print(state.effective_capacity())
# -> {'effective_rank': 28.57, 'max_rank': 32, 'condition_number': 6.98}
```

### 2. Surgical Injection into Hugging Face Models
```python
from transformers import AutoModelForCausalLM
from gist_memory import GistModelAdapter, GistCache

model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

# Graft Gist Memory onto layers 3, 7, 11, 15
adapter = GistModelAdapter(model, target_layers=[3, 7, 11, 15], d_map=32)

# Zero-init guarantee: base model is 100% untouched
assert adapter.verify_zero_init(test_input_ids) == 0.0

# Freeze trunk, train only Gist associative projections (< 1% of total params)
stats = adapter.freeze_backbone()
print(f"Trainable Gist parameters: {stats['trainable_gist_params']:,} ({stats['trainable_pct']:.2f}%)")
```

### 3. Document Memory Freezing (< 100 KB Snapshots)
```python
from gist_memory import GistState

# Ingest a 50,000 token book or code repository into Gist
_, doc_state = layer(long_document, return_state=True)

# Save cognitive snapshot to disk (~66 KB)
doc_state.save("book_memory.pt")

# In a separate session or user prompt, query without re-reading the book:
restored_state = GistState.load("book_memory.pt")
answer, _ = layer(user_question, state=restored_state)
```

---

## ⚙️ Advanced Capabilities

### Multi-Head Gist Memory (`MultiHeadGistLayer`)
Splits the hidden state into $H$ heads, each with an independent sub-manifold and learnable decay half-life:
```python
from gist_memory import MultiHeadGistLayer

multi_gist = MultiHeadGistLayer(
    d_model=2048,
    num_heads=8,
    d_map=16,
    min_half_life=8.0,      # Short-term scratchpad head
    max_half_life=65536.0,  # Long-term permanent fact head
    learnable_decay=True,
)
```

### Chunked SSD Linear Scan (`ChunkedGistLayer`)
Divides long sequences into blocks of size $C=64$ and performs inter-chunk associative scans, reducing memory complexity from $O(L^2)$ to $O(L \cdot C)$:
```python
from gist_memory import ChunkedGistLayer

chunked_gist = ChunkedGistLayer(d_model=1024, d_map=32, chunk_size=64)
# Executes on 64k sequences without out-of-memory errors
out, state = chunked_gist(long_sequence, return_state=True)
```

### C99 Recurrent Engine
For edge, mobile, or Cosmopolitan libc deployment without Python/PyTorch dependencies:
```bash
cd c
make test
```
Header-only implementation located at `c/gist_memory.h`.

---

## 🧪 Verification & Benchmarks

Run the complete automated test suite:
```bash
python run_tests.py
```
```
======================================================================
  GIST MEMORY AUTOMATED TEST SUITE
======================================================================
[PASS] Numerical Equivalence: Parallel vs Streaming (Square Kernel)
[PASS] Numerical Equivalence: Parallel vs Streaming (ReLU2 Kernel)
[PASS] Numerical Equivalence: Parallel vs Streaming (ELU1 Kernel)
[PASS] Chunked SSD vs Full Sequence Equivalence
[PASS] MultiHead: Forward & Shape Validation
[PASS] MultiHead: Gradient Flow & Parameter Updates
[PASS] MultiHead: Streaming Step Equivalence
[PASS] Decay: MultiScale Progression & Positivity
[PASS] Decay: Data-Dependent Contextual Decay
[PASS] State: Diagnostics, Energy & SVD Capacity
[PASS] State: Disk Save & Load Roundtrip
[PASS] Adapter: Strict Zero-Init Baseline Preservation
[PASS] Adapter: GistCache Step-by-Step Propagation
======================================================================
  TEST RESULTS: 13 PASSED | 0 FAILED
======================================================================
```

Run benchmarks:
```bash
python benchmarks/benchmark_speed_mem.py
python benchmarks/benchmark_associative.py
python benchmarks/benchmark_passkey.py
```

---

## 📁 Repository Structure

```
gist-memory/
├── src/gist_memory/
│   ├── __init__.py        # Public API
│   ├── core.py            # GistLayer (Second-Order Quadratic Manifold)
│   ├── multihead.py       # MultiHeadGistLayer with head-specific manifolds
│   ├── decay.py           # MultiScaleDecay & DataDependentDecay
│   ├── chunked.py         # Chunked SSD block-parallel scan (O(L) long-context)
│   ├── state.py           # Typed GistState (SVD capacity, energy, save/load)
│   └── adapter.py         # Universal HF Model Adapter (GistAdapter, GistCache)
├── c/
│   ├── gist_memory.h      # Pure C99 header-only recurrence engine
│   ├── test_gist_c.c      # C99 verification suite
│   └── Makefile           # Build script
├── benchmarks/
│   ├── benchmark_speed_mem.py    # VRAM scaling vs Softmax KV cache
│   ├── benchmark_associative.py  # Multi-hop variable tracking (Delta > 0)
│   └── benchmark_passkey.py      # Passkey needle-in-a-haystack retrieval
├── examples/
│   ├── 01_basic_usage.py         # 10-line standalone layer demo
│   ├── 02_hf_model_patch.py      # Surgical injection into pretrained LLMs
│   └── 03_state_persistence.py   # Document memory freezing (< 100 KB)
├── tests/                        # Comprehensive test cases
├── run_tests.py                  # Zero-dependency test runner
├── pyproject.toml
└── LICENSE                       # Apache-2.0
```

---

## 📜 Citation

If you build upon Gist Memory in your research or applications, please cite:

```bibtex
@software{gist_memory2026,
  author = {batteryphil},
  title = {Gist Memory: Biomimetic Constructive Associative Memory for Language Models},
  year = {2026},
  url = {https://github.com/batteryphil/gist-memory}
}
```

## 📄 License
Licensed under the [Apache License, Version 2.0](LICENSE).
