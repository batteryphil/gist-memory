# Gist Memory Experiments

This folder contains experiments evaluating the real-world capabilities of `GistLayer`. 

## 1. The Universal Connector (`universal_connector/`)

**Goal:** Test if a memory written by one model's activations can be read by completely different LLMs after training only a small connector.

We trained a "Universal Connector" (a 6M-parameter 2-layer MLP) to map Gist memory states into the embedding spaces of different models (Qwen2.5-0.5B and SmolLM-135M).

### The Good: Short-Context Transfer works
For short documents (~150 tokens), compressing the document into 16 Gist memory vectors worked exceptionally well. The frozen target models (even across different model families and tokenizers) could retrieve facts with **~71% accuracy** (compared to 8% random chance), using only the trained connector. 

This proves Gist is a highly effective **dense communication channel** for multi-agent swarms or compressing short static prompts. See `run.py`.

### The Bad: Long-Context Collapse
When we scaled the document length to 1k - 4k tokens and increased the memory capacity to 64 vectors (`run_long.py`), the state decayed and washed out. Accuracy collapsed to random chance (~8%) across all models.

Chunking the documents into 256-token segments (`run_chunked.py`) and passing a sequence of memory states also failed out of the box, proving that standard LLM attention cannot automatically route through dense chunked summaries without deeper architectural changes.

**Conclusion:** Use Gist memory for short-context compression and cross-model communication. Do not use it as an infinite-context RAG replacement.
