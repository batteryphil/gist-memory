#!/usr/bin/env python3
"""
Long-document version (1k / 2k / 4k tokens) of the write-once memory experiment.

Writer: frozen Qwen2.5-0.5B (first 12 layers) -> Gist state (decay=1, so the
state is exactly M = sum_t k_t v_t^T, Z = sum_t k_t; computed directly in O(L))
-> K learned read queries -> K memory vectors.
Targets: Qwen2.5-0.5B (trained jointly with writer), SmolLM-135M (connector only).
Baselines: no context, full text (if it fits the model's context window),
first-K-tokens text, RAG (MiniLM top sentences within a K-token budget),
memory, memory from the wrong document.
"""
import argparse, json, os, random, sys, time
from pathlib import Path

# ROCm: without this, SDPA falls back to the math kernel and materializes L x L attention.
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run import (Connector, load, GistLayer, ATTRS, NAMES, DEV,  # noqa: E402
                 AutoModelForCausalLM, AutoTokenizer)

K, D_MAP, CHUNK_SIZE = 16, 128, 256
LENGTHS = {1024: 8, 2048: 16, 4096: 32}         # target tokens -> number of facts
TRAIN_FIRST, TEST_FIRST = NAMES[:60], NAMES[60:80]
LAST = """Ortiz Becker Novak Haddad Silva Larsen Moreau Kowalski Tanaka Okafor Rossi Jensen
Petrov Nguyen Fischer Dubois Costa Ahmed Lindqvist Mendes Varga Keller Brennan Sato Duarte
Horvat Abbasi Quinn Laurent Nakamura Ferreira Olsen Wagner Romero Kim Murphy Weber Castillo
Holm Ivanova""".split()

_SUBJ = ["The committee", "A neighbor", "The old train", "Our teacher", "The museum guide", "A tired courier",
         "The night shift", "My cousin", "The local paper", "A group of students", "The mechanic",
         "The radio host", "A quiet stranger", "The council", "The head gardener", "A delivery van"]
_VERB = ["discussed", "postponed", "repaired", "ignored", "announced", "painted", "measured", "described",
         "photographed", "cleaned", "moved", "reviewed", "counted", "rebuilt", "praised", "questioned"]
_OBJ = ["the broken fence", "the annual budget", "a set of old maps", "the noisy elevator", "the river path",
        "a stack of letters", "the leaking roof", "the weekend schedule", "the new bridge", "a box of tools",
        "the school play", "the parking rules", "an unusual sculpture", "the library hours", "the market stalls"]
_TAIL = ["before lunch", "without much fuss", "after the storm", "for the third time", "late in the evening",
         "on a cold morning", "despite the rain", "with great care", "at the last minute", "during the break"]


def filler(rng):
    return f"{rng.choice(_SUBJ)} {rng.choice(_VERB)} {rng.choice(_OBJ)} {rng.choice(_TAIL)}."


def make_long(rng, first_names, target_len):
    n_facts = LENGTHS[target_len]
    people = set()
    while len(people) < n_facts:
        people.add(f"{rng.choice(first_names)} {rng.choice(LAST)}")
    facts = []
    for p in people:
        a = rng.choice(list(ATTRS))
        facts.append((p, a, rng.choice(ATTRS[a])))
    n_sent = int(target_len / 11.5)                      # ~11.5 tokens per sentence
    sents = [filler(rng) for _ in range(n_sent - n_facts)]
    for p, a, v in facts:
        sents.insert(rng.randrange(len(sents) + 1), f"{p}'s {a} is {v}.")
    p, a, v = rng.choice(facts)
    return {"sents": sents, "doc": " ".join(sents), "q": f"Question: What is {p}'s {a}?\nAnswer:",
            "ans": " " + v, "attr": a}


class StateWriter(nn.Module):
    def __init__(self, D=896):
        super().__init__()
        self.in_norm = nn.LayerNorm(D, elementwise_affine=False)
        self.gist = GistLayer(d_model=D, d_map=D_MAP, decay=1.0, use_salience_gate=True)
        self.queries = nn.Parameter(torch.randn(K, D_MAP))
        self.out_norm = nn.LayerNorm(D)

    def forward(self, h, mask):
        B, L, _ = h.shape
        C = CHUNK_SIZE
        pad_len = (C - (L % C)) % C
        if pad_len > 0:
            h = torch.nn.functional.pad(h, (0, 0, 0, pad_len))
            mask = torch.nn.functional.pad(mask, (0, pad_len))
        L_padded = h.shape[1]
        n_chunks = L_padded // C
        h_c = h.view(B * n_chunks, C, -1)
        mask_c = mask.view(B * n_chunks, C)
        
        g = self.gist
        x = self.in_norm(h_c.float())
        k = g._apply_kernel(g.map_norm(g.map_proj(x))) * torch.sigmoid(g.salience_gate(x))
        k = k * mask_c.unsqueeze(-1).float()
        v = g.v_proj(x)
        M = torch.einsum("bld,blD->bdD", k, v)
        Z = k.sum(1)
        qk = g._apply_kernel(self.queries)
        num = torch.einsum("kd,bdD->bkD", qk, M)
        den = torch.einsum("kd,bd->bk", qk, Z).unsqueeze(-1).clamp(min=1e-3)
        out = self.out_norm(num / den)
        return out.view(B, n_chunks * K, -1)


def load_source():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16).to(DEV).eval()
    m.model.layers = m.model.layers[:12]
    m.config.num_hidden_layers = 12
    for p in m.parameters():
        p.requires_grad_(False)
    return tok, m


@torch.no_grad()
def src_hidden(src, docs):
    tok, m = src
    enc = tok(docs, return_tensors="pt", padding=True).to(DEV)
    out = m.model(**enc, output_hidden_states=True, use_cache=False)
    return out.hidden_states[12], enc["attention_mask"]


# ─────────── efficient scoring: lm_head only on the tail positions ───────────
TAIL = 12


def build(tok, model, mem, prefixes, answers):
    emb = model.get_input_embeddings()
    Km = 0 if mem is None else mem.shape[1]
    seqs, labs = [], []
    for p, a in zip(prefixes, answers):
        pi = tok(p, add_special_tokens=False)["input_ids"]
        ai = tok(a, add_special_tokens=False)["input_ids"]
        seqs.append(pi + ai); labs.append([-100] * len(pi) + ai)
    T = max(map(len, seqs))
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.tensor([s + [pad] * (T - len(s)) for s in seqs], device=DEV)
    lab = torch.tensor([[-100] * Km + l + [-100] * (T - len(l)) for l in labs], device=DEV)
    att = torch.tensor([[1] * (Km + len(s)) + [0] * (T - len(s)) for s in seqs], device=DEV)
    x = emb(ids)
    if mem is not None:
        x = torch.cat([mem.to(x.dtype), x], 1)
    return x, att, lab


def answer_logprob(model, x, att, lab):
    h = model.model(inputs_embeds=x, attention_mask=att, use_cache=False).last_hidden_state
    W = min(TAIL, x.shape[1] - 1)
    h, tgt = h[:, -W - 1:-1], lab[:, -W:]
    lp = torch.log_softmax(model.lm_head(h).float(), -1)
    m = tgt != -100
    return (lp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1) * m).sum(-1), m.sum(-1)


# ─────────── RAG baseline (MiniLM sentence retrieval) ───────────
class Retriever:
    def __init__(self):
        name = "sentence-transformers/all-MiniLM-L6-v2"
        from transformers import AutoModel
        self.tok = AutoTokenizer.from_pretrained(name)
        self.m = AutoModel.from_pretrained(name).to(DEV).eval()

    @torch.no_grad()
    def embed(self, texts):
        out = []
        for i in range(0, len(texts), 256):
            enc = self.tok(texts[i:i+256], padding=True, truncation=True, return_tensors="pt").to(DEV)
            h = self.m(**enc).last_hidden_state
            msk = enc["attention_mask"].unsqueeze(-1).float()
            out.append(F.normalize((h * msk).sum(1) / msk.sum(1), dim=-1))
        return torch.cat(out)

    def context(self, ex, target_tok, budget):
        q = ex["q"].split("\n")[0].replace("Question: ", "")
        e = self.embed([q] + ex["sents"])
        order = (e[1:] @ e[0]).argsort(descending=True).tolist()
        picked, used = [], 0
        for i in order:
            n = len(target_tok(ex["sents"][i], add_special_tokens=False)["input_ids"])
            if used + n > budget:
                break
            picked.append(i); used += n
        return " ".join(ex["sents"][i] for i in sorted(picked))


# ─────────── train / eval ───────────
def train(writer, conn, tok, model, src, params, steps, bs, lr, seed, log):
    rng = random.Random(seed)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    t0, run, skipped = time.time(), None, 0
    for step in range(1, steps + 1):
        L = rng.choice(list(LENGTHS))
        ex = [make_long(rng, TRAIN_FIRST, L) for _ in range(bs)]
        h, m = src_hidden(src, [e["doc"] for e in ex])
        mem = conn(writer(h, m))
        x, att, lab = build(tok, model, mem, ["\n" + e["q"] for e in ex], [e["ans"] for e in ex])
        lp, n = answer_logprob(model, x, att, lab)
        loss = -(lp / n).mean()
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            skipped += 1; sched.step(); continue
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(gn):
            skipped += 1; opt.zero_grad(set_to_none=True); sched.step(); continue
        opt.step(); sched.step()
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        if step % 100 == 0 or step == steps:
            log(f"    step {step:5d}/{steps}  loss {run:.3f}  skipped {skipped}  ({time.time()-t0:.0f}s)")


@torch.no_grad()
def evaluate(name, tok, model, src, writer, conn, retr, n_eval, log):
    max_ctx = getattr(model.config, "max_position_embeddings", 10**9)
    res = {}
    for L in LENGTHS:
        rng = random.Random(1000 + L)
        exs = [make_long(rng, TEST_FIRST, L) for _ in range(n_eval)]
        mems = torch.cat([conn(writer(*src_hidden(src, [e["doc"] for e in exs[i:i+8]])))
                          for i in range(0, n_eval, 8)])
        doc_toks = sum(len(tok(e["doc"], add_special_tokens=False)["input_ids"]) for e in exs) / n_eval
        fits = doc_toks + 40 <= max_ctx

        def prefix(c, e, i):
            if c == "no_context":       return None, e["q"]
            if c == "full_text":        return None, f"Context: {e['doc']}\n{e['q']}"
            if c == "first_mem_len_tokens":
                ids = tok(e["doc"], add_special_tokens=False)["input_ids"][:K]
                return None, f"Context: {tok.decode(ids)}\n{e['q']}"
            if c == "rag_mem_len_tokens":    return None, f"Context: {retr.context(e, tok, K)}\n{e['q']}"
            if c == "memory":           return mems[i:i+1], "\n" + e["q"]
            if c == "memory_wrong_doc": j = (i + 1) % n_eval; return mems[j:j+1], "\n" + e["q"]

        conds = ["no_context", "full_text", "first_mem_len_tokens", "rag_mem_len_tokens", "memory", "memory_wrong_doc"]
        r = {"doc_tokens": round(doc_toks)}
        for c in conds:
            if c == "full_text" and not fits:
                r[c] = None; continue
            correct = 0
            for i, e in enumerate(exs):
                mem, p = prefix(c, e, i)
                cands = [" " + v for v in ATTRS[e["attr"]]]
                memb = None if mem is None else mem.expand(len(cands), -1, -1)
                x, att, lab = build(tok, model, memb, [p] * len(cands), cands)
                lp, _ = answer_logprob(model, x, att, lab)
                correct += cands[lp.argmax().item()] == e["ans"]
            r[c] = correct / n_eval
        res[L] = r
        log(f"  [{name} @ {L}] " + "  ".join(
            f"{k}={'N/A' if v is None else (v if k == 'doc_tokens' else f'{v*100:.1f}%')}" for k, v in r.items()))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    steps1, steps2, bs, n_eval = (60, 60, 4, 16) if a.quick else (4000, 3000, 8, 150)
    logf = open(HERE / ("log_long_quick.txt" if a.quick else "log_long.txt"), "w")
    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    torch.manual_seed(0)
    log(f"device={DEV} K={K} d_map={D_MAP} steps1={steps1} steps2={steps2} bs={bs} n_eval={n_eval}")
    src = load_source()
    src[0].padding_side = "right"
    retr = Retriever()
    rms = lambda mdl: mdl.get_input_embeddings().weight.float().pow(2).mean().sqrt().item()
    results = {"config": dict(K=K, d_map=D_MAP, steps1=steps1, steps2=steps2, bs=bs, n_eval=n_eval,
                              lengths=LENGTHS)}

    log("\n== Phase 1: writer + connector for qwen0.5b ==")
    tok, model = load("Qwen/Qwen2.5-0.5B")
    writer = StateWriter().to(DEV)
    conn = Connector(896, model.config.hidden_size, rms(model)).to(DEV)
    train(writer, conn, tok, model, src, list(writer.parameters()) + list(conn.parameters()),
          steps1, bs, 5e-4, 1, log)
    writer.eval()
    for p in writer.parameters():
        p.requires_grad_(False)
    torch.save(writer.state_dict(), HERE / "writer_long.pt")
    results["qwen0.5b"] = evaluate("qwen0.5b", tok, model, src, writer, conn, retr, n_eval, log)
    del model; torch.cuda.empty_cache()

    log("\n== Phase 2: connector only (writer frozen) for smollm135m ==")
    tok, model = load("HuggingFaceTB/SmolLM-135M")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    conn = Connector(896, model.config.hidden_size, rms(model)).to(DEV)
    train(writer, conn, tok, model, src, list(conn.parameters()), steps2, bs, 1e-3, 2, log)
    results["smollm135m"] = evaluate("smollm135m", tok, model, src, writer, conn, retr, n_eval, log)

    out = HERE / ("results_long_quick.json" if a.quick else "results_chunked.json")
    out.write_text(json.dumps(results, indent=2))
    log(f"\nSaved {out}")


if __name__ == "__main__":
    main()
