"""
Final numerical auditor for SUMMARY.md. Independently re-derive every number.

numpy only (no scipy). Loads raw per-sample CSVs and recomputes everything.
"""
import os, sys, json, glob, re
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DZ = os.path.join(ROOT, "data_matched_z")
DL = os.path.join(ROOT, "data_matched_largez")
D0 = os.path.join(ROOT, "data_beta0")


def load_dir(d):
    """Return dict keyed by (L,beta) -> structured arrays."""
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
        arr = np.genfromtxt(f, delimiter=",", names=True)
        if arr.size == 0:
            continue
        L = int(arr["L"][0])
        beta = float(arr["beta"][0])
        z = float(arr["z"][0])
        out[(L, beta)] = dict(f=os.path.basename(f), L=L, beta=beta, z=z, a=arr)
    return out


matched = load_dir(DZ)
largez = load_dir(DL)
b0 = load_dir(D0)


def per_sample_mean(a, key="D"):
    """Mean over samples and standard error of the mean."""
    v = a[key]
    n = len(v)
    m = v.mean()
    if n > 1:
        se = v.std(ddof=1) / np.sqrt(n)
    else:
        se = 0.0
    return m, se, n


print("=" * 78)
print("ITEM 1: beta=0 anchor")
print("=" * 78)
baseline = json.load(open(os.path.join(ROOT, "baseline_D.json")))
print("SUMMARY claim: D = 0.288528 0.457099 0.540909 0.595933 0.636754 0.669189")
print("SUMMARY claim: diff = 1.4e-11 4.4e-11 6.2e-12 2.1e-11 4.1e-12 1.7e-11")
print()
for L in (6, 12, 18, 24, 30, 36):
    key = (L, 0.0)
    if key not in b0:
        print(f"  L={L}: MISSING in data_beta0")
        continue
    a = b0[key]["a"]
    Dv = a["D"]
    Dgs = baseline[str(L)]["D"]
    uniq = np.unique(np.round(Dv, 12))
    m, se, n = per_sample_mean(a)
    zval = b0[key]["z"]
    print(f"  L={L:2d} n={n:3d} z_file={zval:8.4f} z_distinct={len(uniq)} "
          f"D_mean={m:.12f} D_gs={Dgs:.12f} diff={abs(m-Dgs):.3e}")
    if len(uniq) > 1:
        print(f"        ** NOT all samples identical: distinct D = {uniq}")
    # also check per-sample max deviation
    maxdev = np.max(np.abs(Dv - Dgs))
    print(f"        max per-sample |D - D_gs| = {maxdev:.3e}")
print()
print("Also: does each sample really hit the same D (spread within file)?")
for L in (6, 12, 18, 24, 30, 36):
    key = (L, 0.0)
    if key not in b0:
        continue
    Dv = b0[key]["a"]["D"]
    print(f"  L={L:2d}: D range = {Dv.min():.12f} .. {Dv.max():.12f}  spread={Dv.max()-Dv.min():.2e}")
