import os, json, glob
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DZ = os.path.join(ROOT, "data_matched_z")
DL = os.path.join(ROOT, "data_matched_largez")
D0 = os.path.join(ROOT, "data_beta0")


def load_dir(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
        arr = np.genfromtxt(f, delimiter=",", names=True)
        if arr.size == 0:
            continue
        L = int(arr["L"][0]); beta = float(arr["beta"][0]); z = float(arr["z"][0])
        out[(L, beta)] = dict(f=os.path.basename(f), L=L, beta=beta, z=z, a=arr)
    return out


matched = load_dir(DZ); largez = load_dir(DL); b0 = load_dir(D0)

def stat(a, key="D"):
    v = a[key]; n = len(v); m = v.mean()
    se = v.std(ddof=1)/np.sqrt(n) if n > 1 else 0.0
    return m, se, n

# rebuild the grid
rows = []
for src, dd in (("matched", matched), ("largez", largez)):
    for (L, beta), v in dd.items():
        D, sD, n = stat(v["a"])
        DV, sDV, _ = stat(v["a"], "DV")
        jm = v["a"]["j"].mean()
        rows.append(dict(src=src, L=L, beta=beta, z=v["z"], D=D, sD=sD, n=n, DV=DV,
                         jm=jm, f=v["f"]))
rows.sort(key=lambda r: (r["L"], r["beta"]))

print("%-8s %3s %8s %9s %11s %11s %5s %7s %s" % ("src","L","beta","z","D","sD","n","<j>","file"))
for r in rows:
    print("%-8s %3d %8.4f %9.4f %11.6f %11.6f %5d %7.3f %s" %
          (r["src"], r["L"], r["beta"], r["z"], r["D"], r["sD"], r["n"], r["jm"], r["f"]))

print()
print("distinct z values:", sorted(set(round(r["z"],3) for r in rows)))
print("n distinct beta:", len(set(r["beta"] for r in rows)))
print("total rows:", len(rows))
json.dump([{k: (v if not isinstance(v, np.floating) else float(v)) for k, v in r.items()}
           for r in rows], open(os.path.join(ROOT, "docs", "_grid.json"), "w"), indent=1)
