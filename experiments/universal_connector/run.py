#!/usr/bin/env python3
"""
Write-once memory, per-model connectors.

A Gist writer compresses a document (via frozen Qwen2.5-0.5B hidden states) into
K memory vectors. Each target LLM gets its own small MLP connector mapping those
vectors to K soft input tokens. The writer is trained only with the first target;
later targets train only their connector, with the writer frozen.

Usage:
  python run.py            # full experiment
  python run.py --quick    # tiny smoke test
"""
import argparse, json, math, os, random, sys, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from gist_memory import GistLayer  # noqa: E402

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path(__file__).resolve().parent

WRITER_MODEL = "Qwen/Qwen2.5-0.5B"
WRITER_LAYER = 12
TARGETS = {
    "qwen0.5b": "Qwen/Qwen2.5-0.5B",
    "smollm135m": "HuggingFaceTB/SmolLM-135M",
    "qwen1.5b-instruct": "Qwen/Qwen2.5-1.5B-Instruct",
}

# ───────────────────────────── synthetic data ─────────────────────────────
NAMES = """Lena Marco Priya Tobias Aisha Henrik Sofia Dmitri Yuki Carlos Nadia Felix Ingrid Omar
Clara Rafael Mei Jonas Leila Victor Hana Mateo Greta Samir Elena Bruno Anya Kofi Lucia Pavel
Rosa Emil Zara Hugo Ines Kenji Maya Oskar Lina Diego Freya Arjun Nora Luca Selma Ivan Alma
Theo Mira Pablo Signe Ravi Ada Nils Eva Tariq Iris Leon Vera Malik Olga Simon June Amir Kira
Felipe Dana Erik Lola Yusuf Tess Aldo Wren Idris Cora Joel Nell Ilya Ruth Gael""".split()
TRAIN_NAMES, TEST_NAMES = NAMES[:60], NAMES[60:80]

ATTRS = {
    "favorite color": "red blue green yellow purple orange pink black white brown gray teal".split(),
    "home city": "Paris Tokyo Cairo Lima Oslo Rome Dublin Seoul Madrid Berlin Vienna Prague".split(),
    "pet": "dog cat parrot rabbit hamster turtle horse goat snake ferret lizard pig".split(),
    "job": "doctor teacher pilot chef lawyer farmer nurse baker painter writer plumber dentist".split(),
}
FILLER = [
    "The weather that week was mild and mostly cloudy.",
    "Several people arrived late to the meeting.",
    "The library on the corner closed early on Sundays.",
    "A new bakery opened near the train station.",
    "Most of the conversation was about the upcoming holiday.",
    "Traffic was heavy on the main road that morning.",
    "Someone had left an umbrella by the door.",
    "The garden needed watering after the dry spell.",
    "There was a long queue at the post office.",
    "Everyone agreed the coffee was too strong.",
    "The old clock in the hall had stopped again.",
    "A small concert was planned for the weekend.",
]


def make_example(rng, names):
    people = rng.sample(names, 4)
    facts = []
    for p in people:
        a = rng.choice(list(ATTRS))
        facts.append((p, a, rng.choice(ATTRS[a])))
    sents = []
    for p, a, v in facts:
        sents += rng.sample(FILLER, rng.randint(1, 2))
        sents.append(f"{p}'s {a} is {v}.")
    sents += rng.sample(FILLER, 1)
    p, a, v = rng.choice(facts)
    return {"doc": " ".join(sents), "q": f"Question: What is {p}'s {a}?\nAnswer:",
            "ans": " " + v, "attr": a}


# ───────────────────────────── modules ─────────────────────────────
class Writer(nn.Module):
    """Gist state over source hidden states, read out by K learned queries."""
    def __init__(self, D=896, d_map=64, K=16):
        super().__init__()
        self.in_norm = nn.LayerNorm(D, elementwise_affine=False)
        self.gist = GistLayer(d_model=D, d_map=d_map, decay=1.0, use_salience_gate=True)
        self.queries = nn.Parameter(torch.randn(K, d_map))
        self.out_norm = nn.LayerNorm(D)

    def forward(self, h, mask):
        h = self.in_norm(h.float())
        _, st = self.gist(h, return_state=True, attention_mask=mask)
        qk = self.gist._apply_kernel(self.queries)                     # [K, d]
        num = torch.einsum("kd,bdD->bkD", qk, st.M)
        den = torch.einsum("kd,bd->bk", qk, st.Z).unsqueeze(-1).clamp(min=1e-3)
        return self.out_norm(num / den)                                 # [B, K, D]


class Connector(nn.Module):
    def __init__(self, D_in, D_out, emb_rms):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D_in, 2048), nn.GELU(), nn.Linear(2048, D_out))
        self.register_buffer("rms", torch.tensor(float(emb_rms)))
        self.gain = nn.Parameter(torch.ones(()))

    def forward(self, m):
        x = self.net(m)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        return x * self.rms * self.gain


# ───────────────────────────── helpers ─────────────────────────────
TOKENIZER_OVERRIDE = {}


def load(name):
    tok = AutoTokenizer.from_pretrained(TOKENIZER_OVERRIDE.get(name, name))
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


@torch.no_grad()
def source_hidden(src_tok, src_model, docs):
    enc = src_tok(docs, return_tensors="pt", padding=True).to(DEV)
    out = src_model(**enc, output_hidden_states=True)
    return out.hidden_states[WRITER_LAYER], enc["attention_mask"]


def build(tok, model, mem, prefixes, answers):
    """inputs_embeds = [mem?] + prefix + answer; labels only on answer tokens."""
    emb = model.get_input_embeddings()
    K = 0 if mem is None else mem.shape[1]
    seqs, labs = [], []
    for p, a in zip(prefixes, answers):
        pi = tok(p, add_special_tokens=False)["input_ids"]
        ai = tok(a, add_special_tokens=False)["input_ids"]
        seqs.append(pi + ai)
        labs.append([-100] * len(pi) + ai)
    T = max(map(len, seqs))
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.tensor([s + [pad] * (T - len(s)) for s in seqs], device=DEV)
    lab = torch.tensor([[-100] * K + l + [-100] * (T - len(l)) for l in labs], device=DEV)
    att = torch.tensor([[1] * (K + len(s)) + [0] * (T - len(s)) for s in seqs], device=DEV)
    x = emb(ids)
    if mem is not None:
        x = torch.cat([mem.to(x.dtype), x], 1)
    return x, att, lab


def answer_logprob(model, x, att, lab):
    logits = model(inputs_embeds=x, attention_mask=att).logits.float()
    lp = torch.log_softmax(logits[:, :-1], -1)
    tgt = lab[:, 1:]
    m = tgt != -100
    tok_lp = lp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1) * m
    return tok_lp.sum(-1), m.sum(-1)


# ───────────────────────────── train / eval ─────────────────────────────
def train(writer, conn, tok, model, src, params, steps, bs, lr, seed, log):
    rng = random.Random(seed)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    t0, run, skipped = time.time(), None, 0
    for step in range(1, steps + 1):
        ex = [make_example(rng, TRAIN_NAMES) for _ in range(bs)]
        h, m = source_hidden(*src, [e["doc"] for e in ex])
        mem = conn(writer(h, m))
        x, att, lab = build(tok, model, mem, ["\n" + e["q"] for e in ex], [e["ans"] for e in ex])
        lp, n = answer_logprob(model, x, att, lab)
        loss = -(lp / n).mean()
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            skipped += 1
            sched.step()
            continue
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(gn):
            skipped += 1
            opt.zero_grad(set_to_none=True)
            sched.step()
            continue
        opt.step(); sched.step()
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        if step % 100 == 0 or step == steps:
            log(f"    step {step:5d}/{steps}  loss {run:.3f}  skipped {skipped}  ({time.time()-t0:.0f}s)")


@torch.no_grad()
def evaluate(name, tok, model, src, writer, conn, n_eval, seed, log):
    rng = random.Random(seed)
    exs = [make_example(rng, TEST_NAMES) for _ in range(n_eval)]
    K = writer.queries.shape[0] if writer is not None else 16
    mems = None
    if writer is not None:
        mems = []
        for i in range(0, n_eval, 32):
            h, m = source_hidden(*src, [e["doc"] for e in exs[i:i+32]])
            mems.append(conn(writer(h, m)))
        mems = torch.cat(mems)

    def prefix(cond, e, i):
        if cond == "no_context":
            return None, e["q"]
        if cond == "full_text":
            return None, f"Context: {e['doc']}\n{e['q']}"
        if cond == "text_same_budget":
            ids = tok(e["doc"], add_special_tokens=False)["input_ids"][:K]
            return None, f"Context: {tok.decode(ids)}\n{e['q']}"
        if cond == "memory":
            return mems[i:i+1], "\n" + e["q"]
        if cond == "memory_wrong_doc":
            return mems[(i + 1) % n_eval:(i + 1) % n_eval + 1], "\n" + e["q"]

    conds = ["no_context", "full_text", "text_same_budget"]
    if writer is not None:
        conds += ["memory", "memory_wrong_doc"]
    res = {}
    for c in conds:
        correct = 0
        for i, e in enumerate(exs):
            mem, p = prefix(c, e, i)
            cands = [" " + v for v in ATTRS[e["attr"]]]
            memb = None if mem is None else mem.expand(len(cands), -1, -1)
            x, att, lab = build(tok, model, memb, [p] * len(cands), cands)
            lp, _ = answer_logprob(model, x, att, lab)
            correct += cands[lp.argmax().item()] == e["ans"]
        res[c] = correct / n_eval
    log(f"  [{name}] " + "  ".join(f"{k}={v*100:.1f}%" for k, v in res.items()))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    steps1, steps2, bs, n_eval = (30, 30, 8, 24) if a.quick else (3000, 2000, 32, 300)

    logf = open(OUT / ("log_quick.txt" if a.quick else "log.txt"), "w")
    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    torch.manual_seed(0)
    log(f"device={DEV}  steps1={steps1} steps2={steps2} bs={bs} n_eval={n_eval}")
    src_tok, src_model = load(WRITER_MODEL)
    src_tok.padding_side = "right"
    src = (src_tok, src_model)
    results = {"config": dict(steps1=steps1, steps2=steps2, bs=bs, n_eval=n_eval,
                              writer_layer=WRITER_LAYER, K=16, d_map=64)}

    def emb_rms(model):
        return model.get_input_embeddings().weight.float().pow(2).mean().sqrt().item()

    # Phase 1: writer + connector for Qwen2.5-0.5B (reuse the already-loaded model)
    log("\n== Phase 1: train writer + connector for qwen0.5b ==")
    writer = Writer().to(DEV)
    tok, model = src_tok, src_model
    conn = Connector(896, model.config.hidden_size, emb_rms(model)).to(DEV)
    train(writer, conn, tok, model, src, list(writer.parameters()) + list(conn.parameters()),
          steps1, bs, 5e-4, 1, log)
    writer.eval()
    for p in writer.parameters():
        p.requires_grad_(False)
    results["qwen0.5b"] = evaluate("qwen0.5b", tok, model, src, writer, conn, n_eval, 123, log)
    torch.save(writer.state_dict(), OUT / "writer.pt")

    # Phase 2/3: frozen writer, new connectors for other models
    for key in ["smollm135m", "qwen1.5b-instruct"]:
        log(f"\n== Connector only (writer frozen): {key} ==")
        tok, model = load(TARGETS[key])
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        conn = Connector(896, model.config.hidden_size, emb_rms(model)).to(DEV)
        train(writer, conn, tok, model, src, list(conn.parameters()), steps2, bs, 1e-3, 2, log)
        results[key] = evaluate(key, tok, model, src, writer, conn, n_eval, 123, log)

        if key == "smollm135m":
            log("\n== Control: SmolLM connector on an UNTRAINED (random) writer ==")
            torch.manual_seed(1)
            rand_writer = Writer().to(DEV).eval()
            for p in rand_writer.parameters():
                p.requires_grad_(False)
            conn = Connector(896, model.config.hidden_size, emb_rms(model)).to(DEV)
            train(rand_writer, conn, tok, model, src, list(conn.parameters()), steps2, bs, 1e-3, 2, log)
            r = evaluate("smollm135m+random_writer", tok, model, src, rand_writer, conn, n_eval, 123, log)
            results["smollm135m_random_writer"] = r
        del model
        torch.cuda.empty_cache()

    path = OUT / ("results_quick.json" if a.quick else "results.json")
    path.write_text(json.dumps(results, indent=2))
    log(f"\nSaved {path}")


if __name__ == "__main__":
    main()
