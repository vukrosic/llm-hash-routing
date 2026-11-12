"""Print table-minus-learned gap and learned churn along training for pairs of runs: python3 curve.py L_name H_name"""
import json, sys
R = {}
for l in open("results.jsonl"): r = json.loads(l); R[r["name"]] = r
L, H = R[sys.argv[1]], R[sys.argv[2]]
print("step  learned  table   gap     churn  share")
for a, b in zip(L["curve"], H["curve"]):
    ch = [a[k] for k in a if k.endswith(".churn")]; sh = [a[k] for k in a if k.endswith(".token")]
    print(f'{a["step"]:5d} {a["web"]:.3f}   {b["web"]:.3f}  {b["web"]-a["web"]:+.3f}  {sum(ch)/len(ch) if ch else 0:.2f}   {sum(sh)/len(sh):.2f}')
print("params", L["params"], "tokens/param", round(L["args"]["steps"] * L["args"]["batch"] * L["args"]["L"] / L["params"], 1))
