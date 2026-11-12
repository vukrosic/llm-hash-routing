"""The load rule for a fixed token->expert table, checked exactly on real token frequencies (no training needed).

Rule: a fixed table cannot split a token, so the k experts of the most common token (share p) each carry p/k of all
routed work, against a fair share of 1/E. The busiest expert therefore carries at least  p * E / k  times the average.
A frequency-balanced table reaches  max(1, p * E / k); a random table is worse. Only the ratio E/k matters.

Frequencies: GPT-2 tokens of FineWeb (counts.npy), and the same text re-tokenized with cl100k_base.
Writes rule.json for the paper figure."""
import json, os
import numpy as np, tiktoken

HERE = os.path.dirname(os.path.abspath(__file__)); rng = np.random.default_rng(0)


def freqs():
    out = {"gpt2 (50K)": np.load(os.path.join(HERE, "data/counts.npy")).astype(float)}
    x = np.memmap(os.path.join(HERE, "data/fineweb_val_000000.bin"), dtype=np.uint16, mode="r")[512:512 + 3_000_000]
    text = tiktoken.get_encoding("gpt2").decode(x.tolist())
    for name, label in [("cl100k_base", "cl100k (100K)")]:
        ids = np.array(tiktoken.get_encoding(name).encode(text, disallowed_special=()))
        out[label] = np.bincount(ids).astype(float)
    return {k: (v[v > 0] / v.sum()) for k, v in out.items()}


def balanced(f, E, k):
    load = np.zeros(E)
    for p in np.sort(f)[::-1][:20000]: ch = np.argpartition(load, k)[:k]; load[ch] += p / k  # tail tokens are tiny
    load += (1 - load.sum()) / E  # spread the rest evenly (each tail token is far below 1/E)
    return load


def random_table(f, E, k):
    load = np.zeros(E)
    for i in range(0, len(f), 20000):
        g = f[i:i + 20000]
        ch = np.argsort(rng.random((len(g), E)), 1)[:, :k] if k > 1 else rng.integers(0, E, (len(g), 1))
        np.add.at(load, ch.reshape(-1), np.repeat(g / k, k))
    return load


ratio = lambda l: l.max() / l.mean()
drop = lambda l, c: np.clip(l / l.sum() - c / len(l), 0, None).sum()  # share of routed work over capacity c x fair

if __name__ == "__main__":
    F = freqs(); res = {"top": {}, "rows": []}
    for name, f in F.items():
        res["top"][name] = float(f.max()); print(f"{name}: top token share {f.max():.4f}, 1/p = {1 / f.max():.1f}")
    print("\ntokenizer        E   k  E/k  floor p*E/k  balanced  random   drop@1.25 bal/rand")
    for name, f in F.items():
        for E, k in [(8, 1), (16, 1), (32, 1), (64, 1), (128, 2), (256, 4), (384, 6), (64, 4), (256, 8), (128, 1), (256, 2), (512, 4), (768, 6), (256, 1)]:
            fl = max(1.0, f.max() * E / k); b = balanced(f, E, k); r = random_table(f, E, k)
            row = dict(tok=name, E=E, k=k, ratio=E / k, floor=fl, balanced=ratio(b), random=ratio(r), drop_b=drop(b, 1.25), drop_r=drop(r, 1.25))
            res["rows"].append(row)
            print(f"{name:14s} {E:4d} {k:3d} {E // k:4d}  {fl:10.2f}  {ratio(b):8.2f}  {ratio(r):6.2f}   {drop(b, 1.25):6.3f} / {drop(r, 1.25):.3f}")
    json.dump(res, open(os.path.join(HERE, "rule.json"), "w"), indent=1)
