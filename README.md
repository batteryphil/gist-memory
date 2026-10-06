# Gist Memory

Decayed linear-attention memory layer for PyTorch, plus a zero-initialized adapter for adding it to Hugging Face decoder models.

> [!NOTE]
> **Status: Experimental, Proven for Short-Context Compression.** 
> While originally designed for infinite context length, our internal experiments have shown that this fixed-size linear attention state cannot reliably compress documents longer than ~1,000 tokens (accuracy collapses to random chance). 
> 
> However, **it is highly effective at compressing short contexts (~150-300 tokens)** and acts as a universal, dense communication channel between different LLMs.

---

## What it is actually good for

Instead of replacing RAG or scaling to infinite document lengths, `GistLayer` is a powerful tool for:

1. **System Prompt & Persona Compression:** You can run a frozen writer model over a 200-token system prompt, compress it into 16 `Gist` memory vectors, and cache them. Instead of paying the compute and token costs to process that persona on every API call, you just prepend the 16 memory tokens. 
2. **Multi-Agent Swarm Communication ("The Universal Connector"):** Our experiments (see `experiments/universal_connector/`) prove that a memory written by a small, cheap model (e.g., Qwen-0.5B) can be seamlessly read by a completely different model family (e.g., SmolLM-135M or a 72B-parameter giant) by training a tiny 6M parameter MLP connector. Small agents can generate a dense "gist" of an environment and hand it directly to a massive reasoning model without passing the raw text.
3. **Very Short-Term Working Memory:** For streaming agents monitoring a continuous chat or log, Gist naturally decays older information. It maintains a constant-size rolling summary of the most recent ~500 tokens without ever blowing up your GPU memory or KV cache.

## How it works

`GistLayer` is kernelized linear attention with exponential decay — the same family as [Katharopoulos et al. 2020](https://arxiv.org/abs/2006.16236) ("Transformers are RNNs"), [RetNet](https://arxiv.org/abs/2307.08621) and [Gated Linear Attention](https://arxiv.org/abs/2312.06635):

```
k_t = φ(RMSNorm(W_k x_t)) · γ_t        γ_t = σ(w_s·x_t + b)   (optional gate)
q_t = φ(RMSNorm(W_q x_t))              φ ∈ {x², ReLU(x)², ELU(x)+1}
v_t = W_v x_t

M_t = λ M_{t-1} + k_t v_tᵀ             state: d_map × D
Z_t = λ Z_{t-1} + k_t                  normalizer: d_map

y_t   = (q_tᵀ M_t) / (q_tᵀ Z_t + ε)
out_t = x_t + σ(W_g x_t + b_g) ⊙ (W_o y_t)          W_o initialized to 0
```

### Properties that hold
- **Fixed-size recurrent state.** `d_map × D + d_map` floats per layer per sequence (e.g. 32×896 fp32 ≈ 112 KB), independent of sequence length.
- **O(d_map·D) per decoded token** once a state exists.
- **Zero-init identity.** Because `W_o = 0`, a freshly wrapped model produces the same logits as the base model.
- **Parallel ≡ streaming.** The parallel and step-by-step paths give the same outputs.
- **State is linear in (k, v).** States from separate streams can be added; a stream's contribution can be subtracted exactly.

### Limitations
- **It does not scale to long contexts.** The linear-attention state acts as a fixed-size bucket. Over ~1,000+ tokens, new information overwrites older information, washing out the facts. Chunking the state does not fix this without deeper architectural changes. Use RAG for long-context retrieval.
- **Fixed state ⇒ lossy.** It cannot replace the KV cache losslessly.
- **`GistLayer` prefill is O(L²)** (it builds an L×L matrix). Use `ChunkedGistLayer` for long sequences.

---

## Install

```bash
git clone https://github.com/batteryphil/gist-memory.git
cd gist-memory
pip install -e .
```

## Usage

```python
import torch
from gist_memory import GistLayer

layer = GistLayer(d_model=512, d_map=32, decay=0.9995)

out, state = layer(torch.randn(1, 1000, 512), return_state=True)          # prefill
out, state = layer(torch.randn(1, 1, 512), state=state, return_state=True) # one decode step

state.save("state.pt")   # ~64 KB for this config
```

Adding to a Hugging Face model:

```python
from transformers import AutoModelForCausalLM
from gist_memory import GistModelAdapter

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
adapter = GistModelAdapter(model, target_layers=[3, 7, 11, 15], d_map=32)

ids = torch.randint(0, model.config.vocab_size, (1, 32))
assert adapter.verify_zero_init(ids) == 0.0   # wrapped model == base model at init

stats = adapter.freeze_backbone()             # only Gist params trainable
```

---

## Experiments

Check out the `experiments/` folder for code proving that the Universal Connector allows different LLMs to read the same Gist memory via a small 6M parameter MLP.

## Tests and benchmarks

```bash
python run_tests.py                              # internal-consistency tests
python benchmarks/benchmark_extreme_limits.py    # sizes, speed, capacity, toy demos
```

See `docs/PERFORMANCE_STUDY.md` for a detailed breakdown of the benchmark results and debunking of prior long-context claims.

## License
Apache-2.0
