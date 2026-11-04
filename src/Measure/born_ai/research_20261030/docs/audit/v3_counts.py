import sys
sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030/docs/audit")
from v3_common import *

pts = load_all()
print("distinct (L,beta):", len(pts))
tot = sum(e["n"] for e in pts.values())
print("total samples:", tot)
# per directory
import glob
for d in DIRS + ("data_beta0",):
    fs = sorted(glob.glob(f"{R}/{d}/*.csv"))
    n = 0
    keys = set()
    for f in fs:
        hdr, a = read_csv(f)
        n += a.shape[0]
        keys.add((int(a[0, 0]), round(float(a[0, 1]), 4)))
    print(d, "files", len(fs), "samples", n, "distinct points", len(keys))
# duplicates
for k, e in sorted(pts.items()):
    if len(e["files"]) > 1:
        print("multi-file", k, e["files"], "n", e["n"], "z", e["z"])
# z uniqueness per key
for k, e in pts.items():
    if len(e["z"]) != 1:
        print("z not unique", k, e["z"])
# per-L counts
from collections import Counter
cL = Counter(k[0] for k in pts)
print("points per L", sorted(cL.items()))
