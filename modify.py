import re

with open("experiments/universal_connector/run_chunked.py", "r") as f:
    content = f.read()

# Change K
content = re.sub(r'K, D_MAP = 64, 128', r'K, D_MAP, CHUNK_SIZE = 16, 128, 256', content)

# Modify StateWriter
old_sw = '''    def forward(self, h, mask):
        g = self.gist
        x = self.in_norm(h.float())
        k = g._apply_kernel(g.map_norm(g.map_proj(x))) * torch.sigmoid(g.salience_gate(x))
        k = k * mask.unsqueeze(-1).float()
        v = g.v_proj(x)
        M = torch.einsum("bld,blD->bdD", k, v)           # same state as GistLayer(return_state=True)
        Z = k.sum(1)
        qk = g._apply_kernel(self.queries)
        num = torch.einsum("kd,bdD->bkD", qk, M)
        den = torch.einsum("kd,bd->bk", qk, Z).unsqueeze(-1).clamp(min=1e-3)
        return self.out_norm(num / den)'''

new_sw = '''    def forward(self, h, mask):
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
        return out.view(B, n_chunks * K, -1)'''

content = content.replace(old_sw, new_sw)

# Change results_long.json to results_chunked.json
content = content.replace("results_long.json", "results_chunked.json")

# In evaluate(), the first_64_tokens baseline and RAG baseline still extract 64 tokens. That's fine. We can keep them or change to 256. 
# But wait, the RAG baseline uses K (which was 64, now 16) to decide budget! 
# Let's change the RAG budget to be `K_tokens = 64` explicitly so we compare against 64 tokens for RAG, 
# or maybe we should compare against n_chunks * 16 memory tokens! 
# Wait, if memory is N*16 tokens, let's keep RAG budget = N*16 tokens!
content = re.sub(
    r't1 = t\[:K\]; l1 = len\(t1\)', 
    r'budget = mem.shape[1]; t1 = t[:budget]; l1 = len(t1)',
    content
)
content = re.sub(
    r'ans_rag = rag.search\(q, docs\[b\], K\)',
    r'ans_rag = rag.search(q, docs[b], mem.shape[1])',
    content
)

# Rename the keys in evaluate print out
content = content.replace('"first_64_tokens"', '"first_mem_len_tokens"')
content = content.replace('"rag_64_tokens"', '"rag_mem_len_tokens"')
content = content.replace('rag_64_tokens=', 'rag_N_tokens=')
content = content.replace('first_64_tokens=', 'first_N_tokens=')

with open("experiments/universal_connector/run_chunked.py", "w") as f:
    f.write(content)
