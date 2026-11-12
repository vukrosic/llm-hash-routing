"""Per design: learned (L), fixed table (H), fixed table with 50% re-assigned every 25 steps (R).
cost per unit churn = (R - H) / churn(R); weight per choice w = share of the FFN width one routed expert holds."""
import json
import numpy as np
R = {}
for l in open("results.jsonl"): r = json.loads(l); R[r["name"]] = r
def churn(r):
    v = [c[k] for c in r["curve"] for k in c if k.endswith(".churn")]; return sum(v) / len(v)
D = [("no shared, top-1, 4 exp", "E4_L", "E4_H", "E4_R", 1), ("no shared, top-1, 8 exp", "base_L", "base_H", "re50_H", 1),
     ("no shared, top-1, 16 exp", "E16_L", "E16_H", "E16_R", 1), ("no shared, top-2", "ns2_L", "ns2_H", "ns2_R", 1/2),
     ("no shared, top-4", "ns4_L", "ns4_H", "ns4_R", 1/4), ("shared, top-1", "sh1_L", "sh1_H", "sh1_R", 1/2),
     ("shared, top-2", "sh2_L", "sh2_H", "sh2_re50_H", 1/4), ("shared, top-4", "sh4_L", "sh4_H", "sh4_R", 1/8)]
rows = []
print(f'{"design":26s} w     learned  table  hash-L  Lchurn | forced: loss  churn  cost/churn  cost/churn/w')
for n, l, h, r, w in D:
    L, H, F = (R[x]["scores"]["web"] for x in (l, h, r)); cl, cf = churn(R[l]), churn(R[r]); s = (F - H) / cf
    rows.append((n, w, H - L, cl, s)); print(f'{n:26s} {w:.3f} {L:.3f}  {H:.3f}  {H-L:+.3f}  {cl:.2f}  |  {F:.3f}  {cf:.2f}   {s:.3f}      {s/w:.3f}')
w = np.array([x[1] for x in rows]); gap = np.array([x[2] for x in rows]); cl = np.array([x[3] for x in rows]); s = np.array([x[4] for x in rows])
print("corr(cost/churn, w) =", round(np.corrcoef(s, w)[0, 1], 3), " fit cost/churn = %.3f * w + %.3f" % tuple(np.polyfit(w, s, 1)))
pred = s * cl
print("predicted gap (cost/churn x learned churn) vs actual (learned - table):")
for x, p in zip(rows, pred): print(f'  {x[0]:26s} predicted {p:.3f}  actual {-x[2]:+.3f}')
print("corr(pred, actual) =", round(np.corrcoef(pred, -gap)[0, 1], 3))
