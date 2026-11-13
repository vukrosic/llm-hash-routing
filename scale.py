"""Move one condition at a time toward a large model and rerun the three main comparisons.
Usage: python3 scale.py SETTING [SETTING ...] [--arms learned,random,sorted]
Each run appends to results.jsonl through hr.py with tag sc_<setting>_<arm>."""
import json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "--moe 16 --topk 1 --noshared --width 32 --layers 2 --V 4000 --batch 16 --steps 200 --L 48"
SET = {  # one knob moved toward large models; later keys override BASE
    "base": "",
    "V16k": "--V 16000",
    "w64": "--width 64",
    "l4": "--layers 4",
    "ctx192": "--L 192 --batch 4",            # longer context, same tokens per step
    "bigbatch": "--batch 64 --steps 50",      # same tokens, 4x batch, 1/4 steps
    "long": "--steps 600",                    # 3x tokens
    "head32": "--heads 1",                    # head size 32 instead of 8
    "lr2x": "--lr 0.04 --elr 0.02",
    "E64": "--moe 64",
    "ds": "--moe 32 --topk 2 --noshared_off",  # shared expert, top-2 of 32 (DeepSeek-like)
    "combo": "--V 16000 --width 64 --layers 4 --moe 32 --topk 2 --noshared_off",
}
ARMS = {"learned": "", "random": "--hash 0,1 --hashtype random", "sorted": "--hash 0,1 --hashtype sorted",
        "so0": "--hash 0,1 --hashtype sorted --unk skip", "so3": "--hash 0,1 --hashtype sorted --unk skip --sharedonly 3",
        "so10": "--hash 0,1 --hashtype sorted --unk skip --sharedonly 10"}


def merge(*parts):
    out, order = {}, []
    for p in parts:
        tok = p.split(); i = 0
        while i < len(tok):
            k = tok[i]; v = tok[i + 1] if i + 1 < len(tok) and not tok[i + 1].startswith("--") else None
            if k not in out: order.append(k)
            out[k] = v; i += 2 if v is not None else 1
    if "--noshared_off" in out: out.pop("--noshared", None); out.pop("--noshared_off"); order = [k for k in order if k in out]
    return [x for k in order if k in out for x in ([k] if out[k] is None else [k, out[k]])]


args = sys.argv[1:]; arms = ["learned", "random", "sorted"]
if "--arms" in args: i = args.index("--arms"); arms = args[i + 1].split(","); args = args[:i] + args[i + 2:]
extra = ""
if "--extra" in args: i = args.index("--extra"); extra = args[i + 1]; args = args[:i] + args[i + 2:]
for s in args:
    for arm in arms:
        tag = f"sc_{s}_{arm}" + (extra.replace(" ", "").replace("--", "_") if extra else "")
        cmd = ["nice", "-n", "20", "python3", os.path.join(HERE, "hr.py")] + merge(BASE, SET[s], ARMS[arm], extra) + ["--tag", tag]
        t = time.time(); subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
        r = json.loads(open(os.path.join(HERE, "results.jsonl")).readlines()[-1])
        if r["name"] != tag: print(tag, "FAILED"); continue
        sc = r["scores"]; f = lambda k: f"{sc[k]:.3f}" if sc.get(k) is not None else "  -  "
        print(f"{tag:28s} web {sc['web']:.3f}  in0-10 {f('web.in.0')} in1000+ {f('web.in.1000')}  load {sc.get('route.0.trainload', 0):.2f}  {r['train_s']:.0f}s", flush=True)
