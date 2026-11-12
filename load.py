"""Expert load under a fixed token->expert table, from real FineWeb GPT-2 token frequencies (no unknown token).
Reports the busiest expert's load divided by the average load (1.0 = perfect balance). Step time in expert-parallel
training follows the busiest expert, so this ratio is the slowdown of the expert part versus perfect balance."""
import numpy as np
c = np.load("data/counts.npy").astype(float); f = c[c > 0] / c.sum(); rng = np.random.default_rng(0)
def random_table(E, k):
    load = np.zeros(E)
    for p in f: load[rng.choice(E, k, replace=False)] += p / k
    return load.max() / load.mean()
def balanced_table(E, k):
    load = np.zeros(E)
    for p in np.sort(f)[::-1]: ch = np.argpartition(load, k)[:k]; load[ch] += p / k
    return load.max() / load.mean()
print(f"top token share {f.max():.3f}")
print("experts  top-k  fair share per expert  random table  balanced table  floor (top token alone)")
for E, k in [(8, 1), (8, 2), (64, 4), (64, 8), (256, 6), (384, 6), (256, 8)]:
    fair = k / E; floor = (f.max() / k) / (1 / E)
    print(f"{E:7d} {k:6d}  {fair:20.4f}  {random_table(E, k):12.2f}  {balanced_table(E, k):14.2f}  {max(1, floor):10.2f}")

# Share of routed token-expert assignments dropped with a capacity cap, for a very large batch (no sampling noise):
# each expert keeps at most cap / E of all assignments.
def loads(E, k, kind):
    load = np.zeros(E)
    if kind == "random":
        for p in f: load[rng.choice(E, k, replace=False)] += p / k
    else:
        for p in np.sort(f)[::-1]: ch = np.argpartition(load, k)[:k]; load[ch] += p / k
    return load
print("\nexperts  top-k  table      dropped at cap 1.0  cap 1.25  cap 2.0")
for E, k in [(64, 1), (64, 4), (256, 8), (384, 6)]:
    for kind in ("random", "balanced"):
        l = loads(E, k, kind); d = [np.clip(l - c / E, 0, None).sum() for c in (1.0, 1.25, 2.0)]
        print(f"{E:7d} {k:6d}  {kind:9s}  {d[0]:17.3f}  {d[1]:8.3f}  {d[2]:7.3f}")
