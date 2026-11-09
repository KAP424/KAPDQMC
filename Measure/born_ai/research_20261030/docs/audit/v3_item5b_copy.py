import os, json, glob, sys
import numpy as np

ROOT = "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030"

base = json.load(open(os.path.join(ROOT, "baseline_D.json")))
ks = np.array(sorted(int(k) for k in base))
vs = np.array([base[str(k)]["D"] for k in ks])
rows = []
for d in ("data_matched_z", "data_matched_largez"):
    for f in sorted(glob.glob(os.path.join(ROOT, d, "*.csv"))):
        a = np.genfromtxt(f, delimiter=",", names=True)
        if a.size == 0:
            continue
        D = a["D"]
        L = int(a["L"][0])
        rows.append(dict(L=L, beta=float(a["beta"][0]), z=float(a["z"][0]),
                         n=int(a.size), mean=float(D.mean()),
                         sem=float(D.std(ddof=1) / np.sqrt(D.size)),
                         Dgs=float(np.interp(np.log(L), np.log(ks), vs)),
                         src=os.path.basename(f)))
for r in rows:
    r["h"] = r["mean"] / r["Dgs"]


def model(L, z, c, g):
    return 1.0 / (1.0 + (z / (c * L)) ** (-g))


def chi2_fit(rows_, fixed_g=None):
    L = np.array([r["L"] for r in rows_]); z = np.array([r["z"] for r in rows_])
    h = np.array([r["h"] for r in rows_]); s = np.array([r["sem"] / r["Dgs"] for r in rows_])

    def chi2(c, g):
        return np.sum(((model(L, z, c, g) - h) / s) ** 2)
    if fixed_g is None:
        best = (np.inf, 0.18, 2.0)
        for g in np.arange(0.6, 4.01, 0.02):
            for c in np.arange(0.05, 0.60, 0.002):
                v = chi2(c, g)
                if v < best[0]:
                    best = (v, c, g)
        return best[1], best[2], best[0]
    best = (np.inf, 0.18)
    for c in np.arange(0.05, 0.60, 0.0005):
        v = chi2(c, fixed_g)
        if v < best[0]:
            best = (v, c)
    return best[1], fixed_g, best[0]


print("=" * 74)
print("TWO-PARAMETER FORM  D_bar = D_gs(L)/(1+(z/(cL))^-g)")
print("SUMMARY: train chi2/dof = 30 (L<=24), blind max|pull| = 10.6")
print("=" * 74)
train = [r for r in rows if r["L"] <= 24]
c, g, chi = chi2_fit(train)
ndof = len(train) - 2
print("  train n=%d  free fit: c=%.4f g=%.3f  chi2=%.1f  chi2/dof=%.2f"
      % (len(train), c, g, chi, chi / ndof))
for tag, (cc, gg, ch) in (("free", (c, g, chi)),):
    ps, ss = [], []
    print("\n  blind prediction L=30,36:")
    print("   %3s %9s %9s %9s %8s %7s" % ("L", "z", "D_pred", "D_meas", "sem", "pull"))
    for r in sorted([r for r in rows if r["L"] in (30, 36)], key=lambda r: (r["L"], r["z"])):
        pred = r["Dgs"] * model(r["L"], r["z"], cc, gg)
        p = (pred - r["mean"]) / r["sem"]
        ps.append(p)
        print("   %3d %9.3f %9.4f %9.4f %8.4f %7.2f" % (r["L"], r["z"], pred, r["mean"], r["sem"], p))
    ps = np.array(ps)
    print("  -> rms pull %.2f  max|pull| %.2f  chi2/dof %.2f"
          % (np.sqrt(np.mean(ps ** 2)), np.abs(ps).max(), np.sum(ps ** 2) / len(ps)))

# include the old production 0.24214 anchor as final_analysis.py did
rows2 = [r for r in rows if not (r["L"] == 36 and abs(r["beta"] - 3.0) < 1e-3)]
rows2.append(dict(L=36, beta=3.0, z=36 * 36 * np.exp(-6), n=0, mean=0.24214,
                  sem=0.00284,
                  Dgs=float(np.interp(np.log(36), np.log(ks), vs)),
                  src="old_anchor"))
for r in rows2:
    r["h"] = r["mean"] / r["Dgs"]
train2 = [r for r in rows2 if r["L"] <= 24]
c2, g2, ch2 = chi2_fit(train2)
print("\n  [variant: L=36 beta=3 replaced by old anchor 0.24214 +- 0.00284]")
print("  train n=%d  c=%.4f g=%.3f chi2/dof=%.2f" % (len(train2), c2, g2, ch2 / (len(train2) - 2)))
ps = []
for r in sorted([r for r in rows2 if r["L"] in (30, 36)], key=lambda r: (r["L"], r["z"])):
    pred = r["Dgs"] * model(r["L"], r["z"], c2, g2)
    ps.append((pred - r["mean"]) / r["sem"])
ps = np.array(ps)
print("  blind rms %.2f max|pull| %.2f  (n=%d)" % (np.sqrt(np.mean(ps ** 2)), np.abs(ps).max(), len(ps)))

print("\n" + "=" * 74)
print("MONOTONICITY per L (item 10 context)  h = D/D_gs")
print("=" * 74)
for L in sorted({r["L"] for r in rows}):
    sel = sorted([r for r in rows if r["L"] == L], key=lambda r: r["z"])
    if len(sel) < 3:
        continue
    hh = np.array([r["h"] for r in sel])
    print("  L=%2d: " % L + " ".join("%6.3f" % v for v in hh)
          + ("   monotone" if all(np.diff(hh) > 0) else "   NON-MONOTONE"))
