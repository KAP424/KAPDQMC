import os, json, glob
import numpy as np

ROOT = r"d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030"

def load(rels):
    rows = []
    for d in rels:
        for f in sorted(glob.glob(os.path.join(ROOT, d, "*.csv"))):
            a = np.genfromtxt(f, delimiter=",", names=True)
            if a.size == 0:
                continue
            D = a["D"]; L = int(a["L"][0])
            rows.append(dict(L=L, beta=float(a["beta"][0]), z=float(a["z"][0]),
                             n=int(a.size), mean=float(D.mean()),
                             sem=float(D.std(ddof=1) / np.sqrt(D.size)),
                             src=d + "/" + os.path.basename(f), a=a))
    return rows

base = json.load(open(os.path.join(ROOT, "baseline_D.json")))
ks = np.array(sorted(int(k) for k in base))
vs = np.array([base[str(k)]["D"] for k in ks])
for r in load(["data_matched_z", "data_matched_largez"]):
    r["Dgs"] = float(np.interp(np.log(r["L"]), np.log(ks), vs))

def model(L, z, c, g):
    return 1.0 / (1.0 + (z / (c * L)) ** (-g))

def chi2_fit(rows_, fixed_g=None):
    L = np.array([r["L"] for r in rows_]); z = np.array([r["z"] for r in rows_])
    h = np.array([r["mean"] / r["Dgs"] for r in rows_])
    s = np.array([r["sem"] / r["Dgs"] for r in rows_])
    def chi2(c, g): return np.sum(((model(L, z, c, g) - h) / s) ** 2)
    if fixed_g is None:
        best = (np.inf, 0.18, 2.0)
        for g in np.arange(0.6, 4.01, 0.02):
            for c in np.arange(0.05, 0.60, 0.002):
                v = chi2(c, g)
                if v < best[0]: best = (v, c, g)
        return best[1], best[2], best[0]
    best = (np.inf, 0.18)
    for c in np.arange(0.05, 0.60, 0.0005):
        v = chi2(c, fixed_g)
        if v < best[0]: best = (v, c)
    return best[1], fixed_g, best[0]

print("=" * 78)
print("EFFECT OF THE DUPLICATE L18_beta1.0000.csv (present in BOTH dirs)")
print("=" * 78)
rows = load(["data_matched_z", "data_matched_largez"])
for r in rows:
    r["Dgs"] = float(np.interp(np.log(r["L"]), np.log(ks), vs))
tr = [r for r in rows if r["L"] <= 24]
c, g, chi = chi2_fit(tr)
print("  WITH duplicate : n=%d  c=%.4f g=%.3f  chi2/dof=%.2f"
      % (len(tr), c, g, chi / (len(tr) - 2)))
tr2 = [r for r in tr if not (r["L"] == 18 and r["src"].startswith("data_matched_largez"))]
c2, g2, chi2 = chi2_fit(tr2)
print("  WITHOUT (n=%d): c=%.4f g=%.3f  chi2/dof=%.2f"
      % (len(tr2), c2, g2, chi2 / (len(tr2) - 2)))

print("\n  chi2/dof vs constant at fixed z=43.85, WITH duplicate:")
sel = [r for r in rows if abs(r["z"] - 43.85) < 0.01]
m = np.array([r["mean"] for r in sel]); s = np.array([r["sem"] for r in sel])
w = 1 / s ** 2; mu = np.sum(w * m) / np.sum(w)
print("    N=%d  Ls=%s  chi2/dof=%.2f" % (len(sel), [r["L"] for r in sel],
                                           np.sum(w * (m - mu) ** 2) / (len(sel) - 1)))
sel2 = [r for r in sel if not r["src"].startswith("data_matched_largez")]
m = np.array([r["mean"] for r in sel2]); s = np.array([r["sem"] for r in sel2])
w = 1 / s ** 2; mu = np.sum(w * m) / np.sum(w)
print("  WITHOUT duplicate: N=%d Ls=%s chi2/dof=%.2f"
      % (len(sel2), [r["L"] for r in sel2],
         np.sum(w * (m - mu) ** 2) / (len(sel2) - 1)))

print()
print("=" * 78)
print("CLAIM: at z=0.089 only 8-29 samples have j>=1")
print("=" * 78)
for r in sorted([r for r in rows if abs(r["z"] - 0.089) < 0.01], key=lambda r: r["L"]):
    j = r["a"]["j"]
    print("  L=%2d n=%4d  n(j>=1)=%3d  n(j>=2)=%3d  frac(j=0)=%.4f"
          % (r["L"], r["n"], int((j >= 1).sum()), int((j >= 2).sum()), float(np.mean(j == 0))))

print()
print("=" * 78)
print("DATA CENSUS")
print("=" * 78)
z = sorted(glob.glob(os.path.join(ROOT, "data_matched_z", "*.csv")))
l = sorted(glob.glob(os.path.join(ROOT, "data_matched_largez", "*.csv")))
b = sorted(glob.glob(os.path.join(ROOT, "data_beta0", "*.csv")))
print("  data_matched_z      : %d csv files" % len(z))
print("  data_matched_largez : %d csv files" % len(l))
print("  data_beta0          : %d csv files" % len(b))
print("  total (L,beta) rows : %d  (SUMMARY says 41 points in data_matched_z)" % (len(z) + len(l)))
names_z = {os.path.basename(f) for f in z}
names_l = {os.path.basename(f) for f in l}
print("  filenames present in BOTH dirs:", sorted(names_z & names_l))
