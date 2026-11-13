"""Toy test bed for hash routing in a standard transformer with mixture-of-experts feed-forward layers.

Model: the same as llm-value-embeddings/ve.py (pre-norm RMSNorm, RoPE, QK-norm, no biases, untied head, Muon on the 2D
block matrices, Adam on the rest), except every feed-forward layer is a DeepSeek-style MoE: 1 shared expert of half the
dense width plus the top-k of E routed experts, each 1/(2k) of the dense width, so the active width equals a dense FFN.
Router score: sqrt(softplus(z)) as in DeepSeek-V4 (or sigmoid as in V3), aux-loss-free bias balancing on learned layers.

Hash routing (DeepSeek-V4's first 3 layers, Roller et al. 2021): in the layers listed in --hash, a fixed table picks the
k experts from the token ID alone. The router still gives the weights of those k experts, as in V4 (--hashw router), or
they are equal (--hashw equal).
Data: FineWeb (GPT-2 tokens, top-V vocabulary) with planted ParaRel facts, each written 2, 8 or 32 times.

usage: python3 hr.py --moe 8 --hash 0 [--width 64] [--layers 4] [--steps 80] [--tag name]
       python3 hr.py --moe 0     (dense FFN, the plain model)
Appends one json line to results.jsonl.
"""
import argparse, collections, json, os, time
import numpy as np, tiktoken, torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data"); RUNS = os.path.join(HERE, "runs")
EOT = 50256; COPIES = (2, 8, 32); FACTSHARE = 0.10

p = argparse.ArgumentParser()
p.add_argument("--moe", type=int, default=8, help="routed experts per layer, 0 = dense FFN")
p.add_argument("--topk", type=int, default=2); p.add_argument("--hash", default="", help="layers with hash routing, e.g. 0 or 0,1")
p.add_argument("--hashw", default="router", choices=["router", "equal"]); p.add_argument("--score", default="sqrtsoftplus", choices=["sqrtsoftplus", "sigmoid"])
p.add_argument("--noshared", action="store_true", help="no shared expert; routed experts take the full active width")
p.add_argument("--rev", type=float, default=0.0, help="share of training rows with reversed word order (same words, different job)")
p.add_argument("--rlr", type=float, default=1.0, help="router learning rate as a multiple of --elr (0 = router frozen at init)")
p.add_argument("--hashtype", default="random", choices=["random", "balanced", "sorted", "cluster", "clusterbal"], help="balanced: greedy assignment so every expert gets equal token frequency; sorted: tokens in frequency order cut into E runs of equal total frequency (an expert holds words of similar frequency); cluster: words grouped by what follows them (k-means on next-word counts), so an expert holds words used alike; clusterbal: words ordered by cluster, then cut into E runs of equal total frequency (grouped by use and balanced)")
p.add_argument("--pairhash", type=int, default=0, help="N > 0: the N most common words get their table experts shifted by a fixed hash of the previous token, so their uses spread over experts by context (still fixed, no learning)")
p.add_argument("--rin", default="hidden", choices=["hidden", "emb"], help="router input: the layer's hidden state, or the token embedding only (a learned per-token router)")
p.add_argument("--cool", type=float, default=0.3, help="share of steps in the final linear learning-rate cooldown")
p.add_argument("--rehash", type=float, default=0.0, help="every 25 steps, give this share of tokens new random experts in hashed layers (forced churn)")
p.add_argument("--freeze", type=int, default=0, help="at this step, turn every learned layer into a fixed table: each token's most used experts so far")
p.add_argument("--unk", default="table", choices=["table", "router", "skip"], help="router: in hashed layers the unknown-word token uses the learned router; skip: in every MoE layer the unknown token uses only the shared expert and is left out of load and balancing (real models have no unknown token)")
p.add_argument("--ctx", action="store_true", help="after training, for each learned layer, replace its routing at test time by each token's most common expert set (counted on 400 training rows) and record the web-loss increase: how much that layer's routing uses context beyond the word")
p.add_argument("--split", type=int, default=0, help="P > 0: write every word as two pieces (word mod P, P + word // P), so no piece names a word on its own and a layer needs context to know the word, as with subword tokens in large models; fact scores are skipped")
p.add_argument("--sharedonly", type=int, default=0, help="N > 0: the N most common words skip the routed experts and use only the shared expert")
p.add_argument("--switch", type=int, default=0, help="S > 0: hashed layers use the table until step S, then their learned router; before S the router is trained to copy the table")
p.add_argument("--distill", type=float, default=1.0, help="weight of the router-copies-table loss before --switch")
p.add_argument("--cap", type=float, default=0.0, help="expert capacity factor: each expert takes at most cap x (tokens x k / E) tokens per batch, later ones are dropped (weight 0, residual passes); 0 = no cap")
p.add_argument("--bspeed", type=float, default=0.01, help="aux-loss-free bias step (DeepSeek-V4 uses 0.001 over far more steps)")
p.add_argument("--width", type=int, default=48); p.add_argument("--layers", type=int, default=4); p.add_argument("--heads", type=int, default=4)
p.add_argument("--V", type=int, default=4000); p.add_argument("--L", type=int, default=48); p.add_argument("--batch", type=int, default=32)
p.add_argument("--steps", type=int, default=60); p.add_argument("--lr", type=float, default=0.02); p.add_argument("--elr", type=float, default=0.01)
p.add_argument("--log", type=int, default=0, help="every N steps, record web loss and routing stats on the validation rows");
p.add_argument("--threads", type=int, default=2); p.add_argument("--tag", default=""); p.add_argument("--seed", type=int, default=0); p.add_argument("--dev", default="cpu")
a = p.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
HASH = [int(x) for x in a.hash.split(",") if x != ""]
DEV = torch.device(a.dev)
LATE = [False]  # True in the last quarter of training (for the late load statistic)


# ---------------- data (same as ve.py) ----------------
class World:
    def __init__(s, V, L, tokens):
        s.enc = tiktoken.get_encoding("gpt2"); rng = np.random.default_rng(0)
        counts = np.load(os.path.join(DATA, "counts.npy"))
        s.keep = np.unique(np.concatenate([np.argsort(counts)[-(V - 2):], [EOT]])); s.V = len(s.keep) + 1; s.unk = s.V - 1
        s.look = np.full(65536, s.unk, np.int64); s.look[s.keep] = np.arange(len(s.keep))
        s.freq = np.zeros(s.V); s.freq[:len(s.keep)] = counts[s.keep]; s.freq[s.unk] = counts.sum() - counts[s.keep].sum()
        kept = set(s.keep.tolist()); E = lambda t: s.enc.encode(" " + t.strip()); s.E = E
        rels = json.load(open(os.path.join(DATA, "facts.json"))); pool = []
        for g in rels:
            for sub, obj in g["pairs"]:
                if all(t in kept for t in E(sub) + E(obj)): pool.append(dict(r=g["r"], s=sub, o=obj, S=g["S"], T=g["T"]))
        pool = [pool[i] for i in rng.permutation(len(pool))]
        n = int(tokens * FACTSHARE / (sum(COPIES) / 3 * 13)); s.facts = pool[:n]
        for i, f in enumerate(s.facts): f["copies"] = COPIES[i % 3]; f["train"] = f["S"][rng.integers(len(f["S"]))]
        s.byrel = {}
        for g in rels: s.byrel[g["r"]] = g["S"]
        docs = [np.array([EOT] + E(f["S"][rng.integers(len(f["S"]))].replace("[X]", f["s"]).replace("[Y]", f["o"])), np.int64)
                for f in s.facts for _ in range(f["copies"])]
        nfact = sum(map(len, docs))
        tr = np.memmap(os.path.join(DATA, "fineweb_train_000001.bin"), dtype=np.uint16, mode="r", offset=1024)
        web = np.asarray(tr[:tokens - nfact + 2 * L]).astype(np.int64); cut = np.flatnonzero(web == EOT)
        alld = docs + (np.split(web, cut[1:]) if len(cut) > 1 else [web])
        st = np.concatenate([alld[i] for i in rng.permutation(len(alld))]); r = st[:len(st) // L * L].reshape(-1, L)
        s.rows = s.look[r[rng.permutation(len(r))]]
        va = np.memmap(os.path.join(DATA, "fineweb_val_000000.bin"), dtype=np.uint16, mode="r", offset=1024)
        s.val = s.look[np.stack([np.asarray(va[x:x + L]) for x in np.linspace(0, len(va) - L - 1, 300).astype(np.int64)]).astype(np.int64)]
        if a.rev: flip = np.random.default_rng(5).random(len(s.rows)) < a.rev; s.rows[flip] = s.rows[flip, ::-1]
        if a.split:  # every word -> two pieces; rows keep their length, so each row holds half as many words
            P = a.split; nv = P + -(-s.V // P)
            two = lambda w: np.stack([w % P, P + w // P], -1).reshape(*w.shape[:-1], -1)
            s.rows = two(s.rows)[:, :L]; s.val = two(s.val)[:, :L]
            s.V = nv + 1; s.unk = nv  # the last ID is never used
            s.freq = np.bincount(s.rows.reshape(-1), minlength=s.V).astype(float)
        s.valrev = np.ascontiguousarray(s.val[:, ::-1])
        s.nfact = nfact; s.ntok = tokens
        print(f"world: V={s.V}, {len(s.rows)} rows, {len(s.facts)} facts ({nfact} tokens, {nfact / tokens:.1%})", flush=True)

    def prompt(s, tmpl, f):
        pre = tmpl[:tmpl.find("[Y]")].replace("[X]", f["s"]).rstrip()
        return s.look[np.array([EOT] + s.enc.encode(pre))], s.look[np.array(s.E(f["o"]))]


# ---------------- model ----------------
def rope(x):
    t, d = x.shape[-2], x.shape[-1]
    f = 1.0 / 10000 ** (torch.arange(0, d, 2, device=x.device).float() / d); ang = torch.outer(torch.arange(t, device=x.device).float(), f); c, sn = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]; return torch.stack([x1 * c - x2 * sn, x1 * sn + x2 * c], -1).flatten(-2)


def uni(*shape, fan_in):  # nn.Linear's default init
    return nn.Parameter(torch.empty(*shape).uniform_(-fan_in ** -0.5, fan_in ** -0.5))


FREQ = None  # set in main() before the model is built
ROWS = None


def hash_table(V, E, k, seed):
    """Each token ID gets k distinct experts, fixed before training. random: drawn at random.
    balanced: tokens in a random order within frequency, each given the k experts with the least frequency so far."""
    g = torch.Generator().manual_seed(1000 + seed)
    if a.hashtype == "random": return torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(V)])
    if a.hashtype == "sorted":
        order = np.argsort(-FREQ[:V], kind="stable"); cum = np.cumsum(FREQ[order]) / FREQ[:V].sum()
        e0 = np.minimum((cum - FREQ[order] / FREQ[:V].sum() / 2) * E, E - 1).astype(np.int64); out = np.zeros((V, k), np.int64)
        out[order] = (e0[:, None] + np.arange(k)[None] * (E // k)) % E; return torch.tensor(out)
    def cut(order):  # walk tokens in this order, cut into E runs of equal total frequency
        cum = np.cumsum(FREQ[order]) / FREQ[:V].sum(); e0 = np.minimum((cum - FREQ[order] / FREQ[:V].sum() / 2) * E, E - 1).astype(np.int64)
        out = np.zeros((V, k), np.int64); out[order] = (e0[:, None] + np.arange(k)[None] * (E // k)) % E; return torch.tensor(out)
    if a.hashtype in ("cluster", "clusterbal"):  # next-word profile of each word (top 256 next words), k-means into E groups
        r = ROWS[:2000]; cur, nxt = r[:, :-1].reshape(-1), r[:, 1:].reshape(-1); top = np.argsort(-FREQ[:V])[:256]; col = np.full(V, -1); col[top] = np.arange(256)
        M = np.zeros((V, 257)); np.add.at(M, (cur, np.where(col[nxt] >= 0, col[nxt], 256)), 1); M = M / np.maximum(M.sum(1, keepdims=True), 1); M = np.sqrt(M)
        rng = np.random.default_rng(seed); C = M[rng.choice(np.argsort(-FREQ[:V])[:E * 4], E, replace=False)]
        for _ in range(15):
            d = ((M[:, None] - C[None]) ** 2).sum(-1); lab = d.argmin(1)
            for e in range(E):
                if (lab == e).any(): C[e] = M[lab == e].mean(0)
        d = ((M[:, None] - C[None]) ** 2).sum(-1)
        if a.hashtype == "clusterbal":
            lab = d.argmin(1); proj = M @ rng.standard_normal(M.shape[1]); return cut(np.lexsort((proj, lab)))  # within a cluster, neighbours by a fixed direction
        return torch.tensor(np.argsort(d, 1)[:, :k])
    load = np.zeros(E); out = np.zeros((V, k), np.int64); jitter = torch.rand(E, generator=g).numpy() * 1e-9
    for t in np.argsort(-FREQ[:V], kind="stable"):
        ch = np.argsort(load + jitter)[:k]; out[t] = ch; load[ch] += FREQ[t]
    return torch.tensor(out)


class MoE(nn.Module):
    def __init__(s, d, f, E, k, V, hashed, layer):
        super().__init__(); s.E, s.k, s.hashed, s.shared = E, k, hashed, not a.noshared
        hs, he = (f // 2, f // (2 * k)) if s.shared else (0, f // k)
        if s.shared: s.sg, s.su, s.sd = nn.Linear(d, hs, bias=False), nn.Linear(d, hs, bias=False), nn.Linear(hs, d, bias=False)
        s.eg, s.eu, s.ed = uni(E, he, d, fan_in=d), uni(E, he, d, fan_in=d), uni(E, d, he, fan_in=he)  # [expert, out, in]
        s.router = nn.Linear(d, E, bias=False); s.register_buffer("bias", torch.zeros(E))
        if hashed: s.register_buffer("tid2eid", hash_table(V, E, k, layer))
        if hashed and a.pairhash:
            com = torch.zeros(V, dtype=torch.bool); com[torch.tensor(np.argsort(-FREQ[:V - 1])[:a.pairhash].copy())] = True; s.register_buffer("common", com)
        if a.sharedonly: top = torch.zeros(V, dtype=torch.bool); top[torch.tensor(np.argsort(-FREQ[:V - 1])[:a.sharedonly].copy())] = True; s.register_buffer("top", top)
        s.register_buffer("cnt", torch.zeros(V, E))  # training-time token -> expert counts (for --freeze)
        s.idx = None; s.force = None  # test-time per-token table (--ctx)
        s.stat = [0.0, 0.0, 0, 0.0, 0]; s.skip = None  # + last-quarter busiest/average sum and count  # training: summed drop share, summed busiest/average load, batches

    def forward(s, h, ids):
        b, t, d = h.shape; hf = h.reshape(-1, d)
        z = s.router(hf if a.rin == "hidden" else F.rms_norm(s.x0, (d,)).reshape(-1, d))
        sc = torch.sqrt(F.softplus(z)) if a.score == "sqrtsoftplus" else torch.sigmoid(z)
        idx = s.tid2eid[ids.reshape(-1)] if s.hashed else (sc + s.bias).topk(s.k, -1).indices
        if s.hashed and a.pairhash:  # common words: shift all k experts by a fixed hash of the previous token (distinct experts stay distinct)
            prev = torch.cat([torch.full_like(ids[:, :1], s.tid2eid.shape[0] - 1), ids[:, :-1]], 1).reshape(-1)
            off = (prev * 2654435761 + 12345) % 1000003 % s.E; c = s.common[ids.reshape(-1)]
            idx = torch.where(c[:, None], (idx + off[:, None]) % s.E, idx)
        if s.force is not None: tab, known = s.force; idx = torch.where(known[ids.reshape(-1)][:, None], tab[ids.reshape(-1)], idx)
        if s.hashed and a.unk == "router": u = ids.reshape(-1) == s.tid2eid.shape[0] - 1; idx = torch.where(u[:, None], (sc + s.bias).topk(s.k, -1).indices, idx)
        w = torch.full_like(idx, 1.0 / s.k, dtype=hf.dtype) if (s.hashed and a.hashw == "equal") else sc.gather(1, idx)
        w = w / w.sum(-1, keepdim=True)
        oh = F.one_hot(idx.reshape(-1), s.E)
        s.skip = (ids.reshape(-1) == s.cnt.shape[0] - 1) if a.unk == "skip" else None
        if a.sharedonly: s.skip = s.top[ids.reshape(-1)] | (s.skip if s.skip is not None else False)
        s.z = z if (s.hashed and a.switch) else None  # the copy loss also shapes the hidden state; detached, the router copied only 37% of tokens
        if s.skip is not None: w = w * ~s.skip[:, None]; oh = oh * ~s.skip.repeat_interleave(s.k)[:, None]
        load = oh.sum(0).float()
        if a.cap:  # tokens in order; an expert's tokens past its capacity are dropped
            C = max(1, int(a.cap * oh.sum().item() / s.E)); keep = ((oh.cumsum(0) * oh).sum(-1) <= C).view_as(idx); w = w * keep
            if s.training: s.stat[0] += (oh.sum(-1).view_as(idx) * ~keep).sum().item() / max(oh.sum().item(), 1)
        if s.training:
            q = (load.max() / load.mean()).item(); s.stat[1] += q; s.stat[2] += 1
            if LATE[0]: s.stat[3] += q; s.stat[4] += 1
        G = torch.zeros_like(sc).scatter(1, idx, w)  # weight 0 for experts not chosen
        act = F.silu(torch.einsum("nd,eod->neo", hf, s.eg)) * torch.einsum("nd,eod->neo", hf, s.eu)  # every expert computed, tiny sizes only
        y = torch.einsum("neo,edo->nd", act * G[..., None], s.ed)
        s.idx = idx.detach()
        if s.training and not s.hashed and a.freeze: s.cnt.index_put_((ids.reshape(-1, 1).expand_as(idx), idx), torch.ones_like(idx, dtype=s.cnt.dtype), accumulate=True)
        if s.shared: y = y + s.sd(F.silu(s.sg(hf)) * s.su(hf))
        return y.view(b, t, d)


class Block(nn.Module):
    def __init__(s, d, H, V, layer):
        super().__init__(); s.H, s.dk = H, d // H
        s.n1, s.n2, s.qn, s.kn = nn.RMSNorm(d), nn.RMSNorm(d), nn.RMSNorm(d // H), nn.RMSNorm(d // H)
        s.q, s.k, s.v, s.o = (nn.Linear(d, d, bias=False) for _ in range(4))
        f = int(8 * d / 3) // 8 * 8
        if a.moe: s.ffn = MoE(d, f, a.moe, a.topk, V, layer in HASH, layer)
        else: s.gate, s.up, s.down = nn.Linear(d, f, bias=False), nn.Linear(d, f, bias=False), nn.Linear(f, d, bias=False); s.ffn = None

    def forward(s, x, ids):
        b, t, d = x.shape; h = s.n1(x)
        sp = lambda z: z.view(b, t, s.H, s.dk).transpose(1, 2)
        y = F.scaled_dot_product_attention(rope(s.qn(sp(s.q(h)))), rope(s.kn(sp(s.k(h)))), sp(s.v(h)), is_causal=True)
        x = x + s.o(y.transpose(1, 2).reshape(b, t, d)); h = s.n2(x)
        return x + (s.ffn(h, ids) if s.ffn is not None else s.down(F.silu(s.gate(h)) * s.up(h)))


class Model(nn.Module):
    def __init__(s, V, d, nl, H):
        super().__init__(); s.emb = nn.Embedding(V, d); nn.init.normal_(s.emb.weight, std=d ** -0.5)
        s.blocks = nn.ModuleList(Block(d, H, V, i) for i in range(nl)); s.norm = nn.RMSNorm(d); s.head = nn.Linear(d, V, bias=False)

    def forward(s, ids):
        x = e = s.emb(ids)
        for blk in s.blocks:
            if blk.ffn is not None: blk.ffn.x0 = e
            x = blk(x, ids)
        return s.head(s.norm(x))


def ns(G, steps=5):
    """Newton-Schulz orthogonalization; a 3D tensor is treated as a batch of matrices (one per expert)."""
    A, B, C = 3.4445, -4.7750, 2.0315; X = G / (G.norm(dim=(-2, -1), keepdim=True) + 1e-7); tr = X.shape[-2] > X.shape[-1]
    if tr: X = X.mT
    for _ in range(steps): M = X @ X.mT; X = A * X + (B * M + C * M @ M) @ X
    return X.mT if tr else X


class Muon(torch.optim.Optimizer):
    """As in ve.py; a 3D expert tensor is updated one expert matrix at a time."""
    def __init__(s, ps, lr): super().__init__(ps, dict(lr=lr))

    @torch.no_grad()
    def step(s):
        for g in s.param_groups:
            for w in g["params"]:
                if w.grad is None: continue
                st = s.state[w]; m = st.setdefault("m", torch.zeros_like(w)); m.mul_(0.95).add_(w.grad)
                w.add_(ns(w.grad + 0.95 * m), alpha=-g["lr"] * max(1, w.shape[-2] / w.shape[-1]) ** 0.5)


# ---------------- eval (same as ve.py, plus routing stats) ----------------
@torch.no_grad()
def answer_lp(m, W, items):
    """Sum log-prob of the answer tokens after each prompt."""
    out = []
    for pr, ans in items:
        x = torch.tensor(np.concatenate([pr, ans]), device=DEV)[None]; lp = F.log_softmax(m(x)[0].float(), -1)
        out.append(float(lp[torch.arange(len(pr) - 1, len(pr) - 1 + len(ans), device=DEV), torch.tensor(ans, device=DEV)].sum()))
    return np.array(out)


@torch.no_grad()
def evaluate(m, W):
    m.eval(); r = {}
    x = torch.tensor(W.val, device=DEV); lg = m(x[:, :-1]); nll = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), x[:, 1:].reshape(-1), reduction="none").view(x.shape[0], -1).cpu()
    r["web"] = float(nll.mean())
    rank = np.argsort(np.argsort(-W.freq)); rk_t = rank[W.val[:, 1:]]
    for lo, hi in ((0, 100), (100, 1000), (1000, 4000), (4000, 10 ** 9)):
        msk = (rk_t >= lo) & (rk_t < hi); r[f"web.tgt.{lo}"] = float(nll.numpy()[msk].mean()) if msk.any() else None
    rk_i = rank[W.val[:, :-1]]  # loss after input words of each frequency band (the word that picked the expert)
    for lo, hi in ((0, 10), (10, 100), (100, 1000), (1000, 10 ** 9)):
        msk = (rk_i >= lo) & (rk_i < hi) & (W.val[:, :-1] != W.unk); r[f"web.in.{lo}"] = float(nll.numpy()[msk].mean()) if msk.any() else None
    if a.moe: r.update(routing(m, W))
    if a.rev:
        xr = torch.tensor(W.valrev, device=DEV); lr_ = m(xr[:, :-1]); r["web.rev"] = float(F.cross_entropy(lr_.reshape(-1, lr_.shape[-1]).float(), xr[:, 1:].reshape(-1)))
    rng = np.random.default_rng(1); rels = sorted(W.byrel)
    for c in (() if a.split else COPIES):
        fs = [f for f in W.facts if f["copies"] == c]
        own = answer_lp(m, W, [W.prompt(f["train"], f) for f in fs])
        new = answer_lp(m, W, [W.prompt(f["T"][0], f) for f in fs])
        oth = answer_lp(m, W, [W.prompt(W.byrel[[q for q in rels if q != f["r"]][rng.integers(len(rels) - 1)]][0], f) for f in fs])
        r[f"facts.stored.{c}"] = float(-own.mean()); r[f"facts.new.{c}"] = float(-new.mean()); r[f"facts.gap.{c}"] = float((own - oth).mean())
    m.train(); return r


def routing(m, W):
    """Per layer, on the validation rows (the last forward pass): the busiest expert's share of choices, and how much the
    token ID alone fixes the expert set. 'token' = share of occurrences of a token (seen 5+ times) that go to that token's
    most common expert set; 1.0 means routing is a function of the token ID, i.e. a learned hash."""
    ids = W.val[:, :-1].reshape(-1); out = {}
    for i, blk in enumerate(m.blocks):
        idx = blk.ffn.idx.cpu().numpy(); sets = [tuple(sorted(r)) for r in idx]
        load = np.bincount(idx.reshape(-1), minlength=a.moe) / idx.size
        def agree(tok):
            by = collections.defaultdict(list)
            for t, st in zip(tok, sets): by[t].append(st)
            hit = tot = 0
            for st in by.values():
                if len(st) >= 5: hit += collections.Counter(st).most_common(1)[0][1]; tot += len(st)
            return hit / max(tot, 1)
        out[f"route.{i}.maxload"] = float(load.max()); out[f"route.{i}.token"] = agree(ids)
        out[f"route.{i}.null"] = agree(np.random.default_rng(2).permutation(ids))  # chance level: same counts, token IDs shuffled
    return out


@torch.no_grad()
def ctx_cost(m, W):
    """Test-time context ablation. Count each token's expert sets on 400 training rows, then for one learned layer at a time
    (and for all learned layers together) route every token seen 3+ times (not the unknown token) to its most common set."""
    m.eval(); out = {}
    x = torch.tensor(np.concatenate([W.val, W.valrev]) if a.rev else W.val, device=DEV)  # with --rev, half the scored rows are reversed
    def web(): lg = m(x[:, :-1]); return float(F.cross_entropy(lg.reshape(-1, W.V).float(), x[:, 1:].reshape(-1)))
    base = web(); out["ctx.base"] = base
    tr = torch.tensor(W.rows[:400], device=DEV)[:, :-1]; m(tr); ids = tr.reshape(-1)
    tabs = {}
    for i, blk in enumerate(m.blocks):
        f = blk.ffn
        if f.hashed: continue
        sets = f.idx.sort(-1).values; key = (sets * (f.E ** torch.arange(f.k, device=DEV))).sum(-1)  # one code per expert set
        tab = torch.zeros(W.V, f.k, dtype=torch.long, device=DEV); known = torch.zeros(W.V, dtype=torch.bool, device=DEV)
        for t in torch.unique(ids).tolist():
            if t == W.unk: continue
            msk = ids == t
            if msk.sum() < 3: continue
            ks, c = torch.unique(key[msk], return_counts=True); best = ks[c.argmax()]
            tab[t] = sets[msk][key[msk] == best][0]; known[t] = True
        tabs[i] = (tab, known)
    m(x[:, :-1]); learned = {i: m.blocks[i].ffn.idx.sort(-1).values.clone() for i in tabs}
    for i in tabs:
        f = m.blocks[i].ffn; f.force = tabs[i]; out[f"ctx.{i}"] = web() - base
        out[f"ctx.moved.{i}"] = float((f.idx.sort(-1).values != learned[i]).any(-1).float().mean())
        rnd = hash_table(W.V, f.E, f.k, 77 + i).to(DEV); f.force = (rnd, tabs[i][1]); out[f"ctx.rand.{i}"] = web() - base  # control: random table, same tokens
        f.force = None
    for i in tabs: m.blocks[i].ffn.force = tabs[i]
    out["ctx.all"] = web() - base
    for i in tabs: m.blocks[i].ffn.force = None
    m.train(); return out


# ---------------- train ----------------
def main():
    global FREQ, ROWS
    W = World(a.V, a.L, a.steps * a.batch * a.L); FREQ = W.freq; ROWS = W.rows; m = Model(W.V, a.width, a.layers, a.heads).to(DEV)
    mats = [w for n, w in m.named_parameters() if w.ndim >= 2 and n.startswith("blocks") and "router" not in n]
    rout = [w for n, w in m.named_parameters() if "router" in n]
    rest = [w for w in m.parameters() if all(w is not z for z in mats + rout)]
    groups = [dict(params=rest)] + ([dict(params=rout, lr=a.elr * a.rlr)] if rout else [])
    opts = [Muon(mats, a.lr), torch.optim.Adam(groups, lr=a.elr, betas=(0.9, 0.95))]
    base = [[g["lr"] for g in o.param_groups] for o in opts]
    nparam = sum(w.numel() for w in m.parameters())
    nact = nparam - (sum(blk.ffn.eg.numel() * 3 for blk in m.blocks) * (1 - a.topk / a.moe) if a.moe else 0)
    curve = []; t0 = time.time(); paused = 0.0

    prev_idx = {}

    @torch.no_grad()
    def snap(step):
        m.eval(); x = torch.tensor(W.val, device=DEV); lg = m(x[:, :-1])
        c = dict(step=step, web=round(float(F.cross_entropy(lg.reshape(-1, W.V).float(), x[:, 1:].reshape(-1))), 4))
        if a.moe:
            c.update({k: round(v, 4) for k, v in routing(m, W).items()})
            for i, blk in enumerate(m.blocks):  # churn: share of validation positions whose expert set changed since the last snapshot
                cur = blk.ffn.idx.sort(-1).values
                if i in prev_idx: c[f"route.{i}.churn"] = round(float((cur != prev_idx[i]).any(-1).float().mean()), 4)
                prev_idx[i] = cur
        m.train(); curve.append(c)
    for step in range(a.steps):
        LATE[0] = step >= 0.75 * a.steps
        if a.log and step % a.log == 0 and step: tp = time.time(); snap(step); paused += time.time() - tp
        if a.freeze and step == a.freeze:
            for blk in m.blocks:
                f = blk.ffn
                if f.hashed: continue
                tab = f.cnt.topk(f.k, -1).indices; unseen = f.cnt.sum(-1) == 0
                tab[unseen] = hash_table(f.cnt.shape[0], f.E, f.k, 99)[unseen.cpu()].to(tab.device)
                f.register_buffer("tid2eid", tab); f.hashed = True
        if a.rehash and step and step % 25 == 0:
            for blk in m.blocks:
                f = blk.ffn
                if f.hashed:
                    g = torch.Generator().manual_seed(step); msk = torch.rand(f.tid2eid.shape[0], generator=g) < a.rehash
                    f.tid2eid[msk] = torch.stack([torch.randperm(f.E, generator=g)[:f.k] for _ in range(int(msk.sum()))]).to(f.tid2eid.device) if msk.any() else f.tid2eid[msk]
        sc = min((step + 1) / 20, 1.0, (a.steps - step) / (a.cool * a.steps))
        for o, bl in zip(opts, base):
            for g, lr in zip(o.param_groups, bl): g["lr"] = lr * sc
        x = torch.tensor(W.rows[step * a.batch:(step + 1) * a.batch], device=DEV)
        loss = F.cross_entropy(m(x[:, :-1]).reshape(-1, W.V).float(), x[:, 1:].reshape(-1))
        if a.switch and step < a.switch:  # router learns to copy the table, so the switch moves few words
            for blk in m.blocks:
                f = blk.ffn
                if f.z is not None: tgt = f.tid2eid[x[:, :-1].reshape(-1)]; loss = loss + a.distill * sum(F.cross_entropy(f.z.float(), tgt[:, j]) for j in range(f.k)) / f.k
        if a.switch and step == a.switch:
            for blk in m.blocks:
                if blk.ffn.hashed: blk.ffn.hashed = False; blk.ffn.z = None
        loss.backward()
        for o in opts: o.step()
        m.zero_grad(set_to_none=True)
        for blk in m.blocks:  # aux-loss-free balancing: raise the bias of under-used experts (learned layers only)
            if a.moe and (not blk.ffn.hashed or a.unk == "router"):
                ix = blk.ffn.idx if blk.ffn.skip is None else blk.ffn.idx[~blk.ffn.skip]
                load = torch.bincount(ix.reshape(-1), minlength=a.moe).float(); blk.ffn.bias += a.bspeed * torch.sign(load.mean() - load)
    if DEV.type == "mps": torch.mps.synchronize()
    train_s = time.time() - t0 - paused
    if a.log: snap(a.steps)
    r = evaluate(m, W)
    if a.moe:
        for i, blk in enumerate(m.blocks): d, l, n, ll, ln = blk.ffn.stat; r[f"route.{i}.drop"] = d / max(n, 1); r[f"route.{i}.trainload"] = l / max(n, 1); r[f"route.{i}.lateload"] = ll / max(ln, 1)
    if a.moe and a.ctx: r.update(ctx_cost(m, W))
    name = a.tag or (f"moe{a.moe}k{a.topk}_hash{a.hash.replace(',', '') or 'none'}_{a.hashw}_{a.score}" if a.moe else "dense") + f"_w{a.width}_l{a.layers}_s{a.steps}" + (f"_seed{a.seed}" if a.seed else "")
    res = dict(name=name, args=vars(a), params=nparam, active=int(nact), train_s=round(train_s, 2), scores=r, curve=curve)
    with open(os.path.join(HERE, "results.jsonl"), "a") as f: f.write(json.dumps(res) + "\n")
    show = " ".join(f"{k}={r.get(k, float('nan')):.3f}" for k in ("web", "facts.stored.32", "facts.new.32", "facts.gap.32", "facts.stored.2"))
    rt = " ".join(f"L{i}: token {r[f'route.{i}.token']:.2f} (chance {r[f'route.{i}.null']:.2f}) maxload {r[f'route.{i}.maxload']:.2f}" for i in range(a.layers)) if a.moe else ""
    if a.ctx: rt += "\n  ctx cost: " + " ".join(f"L{i} {r[f'ctx.{i}']:+.3f}" for i in range(a.layers) if f"ctx.{i}" in r) + f"  all {r['ctx.all']:+.3f}  moved " + " ".join(f"{r[f'ctx.moved.{i}']:.2f}" for i in range(a.layers) if f"ctx.moved.{i}" in r) + "  random-table cost " + " ".join(f"{r[f'ctx.rand.{i}']:+.3f}" for i in range(a.layers) if f"ctx.rand.{i}" in r)
    print(f"{name}: params {nparam / 1e6:.2f}M (active {nact / 1e6:.2f}M), train {train_s:.1f}s\n  {show}\n  {rt}", flush=True)


if __name__ == "__main__": main()
