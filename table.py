"""Print learned vs hash pairs: python3 table.py tag1 tag2 ... (arms are <tag>_L and <tag>_H)."""
import json, sys
R = {}
for l in open("results.jsonl"): r = json.loads(l); R[r["name"]] = r
print(f'{"arm":12s} learned  hash    diff   churn(L0,L1 mean over training)  share  maxload(L/H)')
for t in sys.argv[1:]:
    L, H = R.get(t + "_L"), R.get(t + "_H")
    if not L or not H: print(t, "missing"); continue
    sl, sh = L["scores"], H["scores"]; nl = sum(1 for k in sl if k.endswith(".token"))
    ch = [c.get(f"route.{i}.churn") for c in L["curve"] for i in range(nl) if c.get(f"route.{i}.churn") is not None]
    print(f'{t:12s} {sl["web"]:.3f}  {sh["web"]:.3f}  {sh["web"]-sl["web"]:+.3f}   {sum(ch)/max(len(ch),1):.2f}   '
          f'{sum(sl[f"route.{i}.token"] for i in range(nl))/nl:.2f}  {max(sl[f"route.{i}.maxload"] for i in range(nl)):.2f}/{max(sh[f"route.{i}.maxload"] for i in range(nl)):.2f}')
