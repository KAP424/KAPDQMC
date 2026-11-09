import os, json, glob
import numpy as np

ROOT = r"d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030"
base = json.load(open(os.path.join(ROOT, "baseline_D.json")))
ks = np.array(sorted(int(k) for k in base))
vs = np.array([base[str(k)]["D"] for k in ks])

rows = []
for d in ("data_matched_z", "data_matched_largez"):
    for f in sorted(glob.glob(os.path.join(ROOT, d, "*.csv"))):
        a = np.genfromtxt(f, delimiter=",", names=True)
        if a.size == 0:
            continue
        D = a["D"]; L = int(a["L"][0])
        rows.append(dict(L=L, beta=float(a["beta"][0]), z=float(a["z"][0]),
                         mean=float(D.mean()),
                         sem=float(D.std(ddof=1) / np.sqrt(D.size)),
                         src=d + "/" + os.path.basename(f)))

print("=" * 78)
print("'two z deviate from a constant at 2.2-3.3 sigma' -- per-row max pull")
print("=" * 78)
groups = {}
for r in rows:
    groups.setdefault(round(r["z"], 3), []).append(r)
for zt in (0.089, 0.357, 0.803, 1.428, 3.213):
    sel = groups[zt]
    m = np.array([r["mean"] for r in sel]); s = np.array([r["sem"] for r in sel])
    w = 1 / s ** 2; mu = np.sum(w * m) / np.sum(w)
    pull = (m - mu) / s
    chi2 = np.sum(pull ** 2); dof = len(sel) - 1
    print("  z=%7.3f  chi2/dof=%.2f  max|pull|=%.2f  pulls=%s"
          % (zt, chi2 / dof, np.abs(pull).max(),
             " ".join("%+.1f" % p for p in pull)))

print()
print("=" * 78)
print("CENSUS at this moment")
print("=" * 78)
for d in ("data_matched_z", "data_matched_largez", "data_beta0", "figures", "docs",
          "docs/tables", "docs/figures", "docs/audit", "__pycache__"):
    p = os.path.join(ROOT, d)
    n = len(os.listdir(p)) if os.path.isdir(p) else None
    print("  %-22s %s" % (d, ("MISSING" if n is None else "%d entries" % n)))
