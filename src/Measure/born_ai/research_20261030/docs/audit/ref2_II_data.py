"""Referee (II)-d: data-level test of the volume-fraction law S2(R)/z^2 -> 4 f(1-f), D/z^2 -> 16/81, at small z."""
import sys; sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030/docs/audit")
import numpy as np, ref2_common as c
pts, base = c.load()
theory = {"SA": 8 / 9, "SB": 56 / 81, "SC": 8 / 9, "D": 16 / 81}
zs = [0.0890, 0.3570, 0.8030]
for z0 in zs:
    sel = sorted([p for p in pts.values() if abs(p["z"] - z0) / z0 < 5e-3], key=lambda p: p["L"])
    print("z=%.3f  L=%s  n=%s" % (z0, [p["L"] for p in sel], [p["n"] for p in sel]))
    for col in ("SA", "SB", "SC", "D"):
        m = np.array([p["cols"][col].mean() / p["z"] ** 2 for p in sel]); e = np.array([p["cols"][col].std(ddof=1) / np.sqrt(p["n"]) / p["z"] ** 2 for p in sel])
        mu, se = c.wmean(m, e); chi = (((m - mu) / e) ** 2).sum()
        print("   %s: pooled %.4f +- %.4f ; theory(leading) %.4f ; ratio %.3f +- %.3f ; chi2 across L %.1f/%d ; per-L ratio: %s" % (
            col, mu, se, theory[col], mu / theory[col], se / theory[col], chi, len(sel) - 1, np.round(m / theory[col], 2)))
# fraction of samples with j>=1 and share of D from j>=1 at z=0.089
for z0 in (0.0890,):
    for p in sorted([p for p in pts.values() if abs(p["z"] - z0) / z0 < 5e-3], key=lambda p: p["L"]):
        j = p["cols"]["j"]; D = p["cols"]["D"]
        print("  z=0.089 L=%d: frac(j>=1)=%.3f, share of sum(D) from j>=1 = %.2f, mean D | j=0 = %.2e (z^2*16/81 = %.2e), mean D | j>=1 = %.2e" % (
            p["L"], (j >= 1).mean(), D[j >= 1].sum() / D.sum(), D[j == 0].mean(), 16 / 81 * p["z"] ** 2, D[j >= 1].mean() if (j >= 1).any() else np.nan))
