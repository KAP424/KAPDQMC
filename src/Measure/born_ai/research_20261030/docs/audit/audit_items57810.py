import os, json, glob
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DZ = os.path.join(ROOT, "data_matched_z")
DL = os.path.join(ROOT, "data_matched_largez")


def load_dir(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
        arr = np.genfromtxt(f, delimiter=",", names=True)
        if arr.size == 0:
            continue
        L = int(arr["L"][0]); beta = float(arr["beta"][0]); z = float(arr["z"][0])
        out[(L, beta)] = dict(f=os.path.basename(f), L=L, beta=beta, z=z, a=arr)
    return out


rows = []
for d, src in ((DZ, "matched"), (DL, "largez")):
    for (L, beta), v in load_dir(d).items():
        D = v["a"]["D"]
        rows.append(dict(src=src, L=L, beta=beta, z=v["z"], n=len(D),
                         mean=float(D.mean()),
                         sem=float(D.std(ddof=1) / np.sqrt(len(D))),
                         f=v["f"]))
baseline = json.load(open(os.path.join(ROOT, "baseline_D.json")))
ks = np.array(sorted(int(k) for k in baseline))
vs = np.array([baseline[str(k)]["D"] for k in ks])
for r in rows:
    r["Dgs"] = float(np.interp(np.log(r["L"]), np.log(ks), vs))

# ------------------------------------------------- ITEM 7: chi2/dof vs constant
print("=" * 78)
print("ITEM 7: chi2/dof against a constant, per fixed-z row")
print("=" * 78)
print("SUMMARY: z=0.089: 0.91  z=0.357: 2.25  z=0.803: 1.22  z=1.428: 3.25")
print()
groups = {}
for r in rows:
    groups.setdefault(round(r["z"], 3), []).append(r)
for zt in sorted(groups):
    sel = groups[zt]
    if len(sel) < 2:
        continue
    m = np.array([r["mean"] for r in sel]); s = np.array([r["sem"] for r in sel])
    w = 1 / s ** 2
    mu = np.sum(w * m) / np.sum(w)
    chi2 = np.sum(w * (m - mu) ** 2)
    dof = len(sel) - 1
    spread = 100 * (m.max() - m.min()) / m.min()
    print("  z=%8.3f N=%d  chi2/dof=%5.2f  spread=%6.1f%%  mean=%.4f  Ls=%s"
          % (zt, len(sel), chi2 / dof, spread, mu, sorted(r["L"] for r in sel)))

# check duplicate concern: the same (L,beta) appears in both dirs at z=43.85
print("\n  duplicate check -- identical (L,beta) in two dirs:")
for r in rows:
    for q in rows:
        if r is q:
            continue
        if r["L"] == q["L"] and abs(r["beta"] - q["beta"]) < 1e-6 and r["src"] != q["src"]:
            print("   ", r["f"], r["src"], "<->", q["f"], q["src"],
                  "same? D %.6f vs %.6f" % (r["mean"], q["mean"]))
# check for near-duplicate z at same L (18, 1.0)
print("\n  z=43.85 rows at L=18 (two files):")
for r in rows:
    if r["L"] == 18 and abs(r["z"] - 43.85) < 0.01:
        print("   ", r["f"], r["src"], "D=%.6f sem=%.6f n=%d" % (r["mean"], r["sem"], r["n"]))

# ------------------------------------------------- ITEM 5: blind prediction
print()
print("=" * 78)
print("ITEM 5: blind prediction, train L<=24, predict L=30,36")
print("=" * 78)
print("SUMMARY: rms pull 1.21, max|pull| 2.13, chi2/dof 1.47")
print()
ps = []
print("  %3s %9s %9s %9s %8s %7s" % ("L", "z", "D_pred", "D_meas", "sem", "pull"))
for r in sorted([r for r in rows if r["L"] > 24], key=lambda r: (r["L"], r["z"])):
    ref = [q for q in rows if q["L"] <= 24 and abs(q["z"] - r["z"]) < 1e-3]
    if not ref:
        continue
    w = np.array([1 / q["sem"] ** 2 for q in ref])
    m = np.array([q["mean"] for q in ref])
    pred = float(np.sum(w * m) / np.sum(w))
    sepred = float(1 / np.sqrt(w.sum()))
    p = (pred - r["mean"]) / np.hypot(sepred, r["sem"])
    ps.append(p)
    print("  %3d %9.4f %9.4f %9.4f %8.4f %7.2f   (nref=%d)"
          % (r["L"], r["z"], pred, r["mean"], r["sem"], p, len(ref)))
ps = np.array(ps)
print("\n  N=%d  rms pull %.3f  max|pull| %.3f  chi2/dof %.3f"
      % (len(ps), np.sqrt(np.mean(ps ** 2)), np.abs(ps).max(), np.sum(ps ** 2) / len(ps)))

# same test but excluding the duplicated largez file at L=18 (train side)
print("\n  [variant: use only data_matched_z as training set]")
ps2 = []
for r in sorted([r for r in rows if r["L"] > 24], key=lambda r: (r["L"], r["z"])):
    ref = [q for q in rows if q["L"] <= 24 and abs(q["z"] - r["z"]) < 1e-3]
    if not ref:
        continue
    w = np.array([1 / q["sem"] ** 2 for q in ref]); m = np.array([q["mean"] for q in ref])
    pred = float(np.sum(w * m) / np.sum(w)); sepred = float(1 / np.sqrt(w.sum()))
    ps2.append((pred - r["mean"]) / np.hypot(sepred, r["sem"]))
ps2 = np.array(ps2)
print("  N=%d  rms %.3f  max %.3f  chi2/dof %.3f"
      % (len(ps2), np.sqrt(np.mean(ps2 ** 2)), np.abs(ps2).max(), np.sum(ps2 ** 2) / len(ps2)))

# ------------------------------------------------- ITEM 10: h-1 non-monotonic
print()
print("=" * 78)
print("ITEM 10: h-1 non-monotone at L=18 and L=24")
print("=" * 78)
print("SUMMARY L=18: 1.034, 1.057, 1.026 ; L=24: 1.020, 1.009, 1.035")
print("  (h = D/D_gs at large z)")
for L in (12, 18, 24, 30, 36):
    sel = sorted([r for r in rows if r["L"] == L and r["z"] > 4],
                 key=lambda r: r["z"])
    print("  L=%2d:" % L)
    for r in sel:
        print("     z=%9.4f  D=%.5f +- %.5f  h-1=%+.4f +- %.4f"
              % (r["z"], r["mean"], r["sem"], r["mean"] / r["Dgs"] - 1,
                 r["sem"] / r["Dgs"]))

# ------------------------------------------------- ITEM 8: annealed volume law
print()
print("=" * 78)
print("ITEM 8: annealed volume law 2w(L-2w)ln2, w=L/3")
print("=" * 78)
print("SUMMARY: L=12,18,24,36,60 -> 22.2 / 49.9 / 88.7 / 199.6 / 554.5")
for L in (12, 18, 24, 36, 60):
    w = L / 3
    v = 2 * w * (L - 2 * w) * np.log(2)
    print("  L=%2d w=%.1f  S = 2*%.1f*(%d-%.1f)*ln2 = %.2f" % (L, w, w, L, 2 * w, v))
print("  check intermediate sizes:")
for L in (3, 6, 9, 30, 42, 45, 48):
    w = L / 3
    print("  L=%2d -> %.2f" % (L, 2 * w * (L - 2 * w) * np.log(2)))
