"""Run a list of arms one at a time (foreground) and print a table. Usage: python3 sweep.py <name> 'tag|args' ..."""
import json, subprocess, sys
BASE = "--width 32 --layers 2 --steps 200 --batch 16 --log 100"
for spec in sys.argv[1:]:
    tag, extra = spec.split("|")
    out = subprocess.run(["nice", "-n", "20", sys.executable, "hr.py", *BASE.split(), *extra.split(), "--tag", tag], capture_output=True, text=True)
    line = [l for l in out.stdout.splitlines() if "train" in l]; print(line[0] if line else out.stderr[-400:], flush=True)
