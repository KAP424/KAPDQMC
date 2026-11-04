"""Referee (III): is 'undetermined' honest?  Slope ratio r = (dD_bar/dlnL)/(dD_gs/dlnL) under alternative analysis choices."""
import sys; sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030/docs/audit")
import numpy as np, ref2_common as c
pts, base = c.load()
rng = np.random.default_rng(31)
def sel(beta, Lmin, Lmax=99):
    return sorted([p for p in pts.values() if abs(p["beta"] - beta) < 2e-3 and Lmin <= p["L"] <= Lmax], key=lambda p: p["L"])
def ratio(ps, weighted=True, mode="D"):
    x = np.log([p["L"] for p in ps]); gs = np.array([p["Dgs"] for p in ps])
    y = np.array([p["mean"] for p in ps]); s = np.array([p["sem"] for p in ps])
    if mode == "h": y, s = y / gs, s / gs
    w = 1 / s ** 2 if weighted else np.ones_like(s)
    X = np.column_stack([np.ones_like(x), x]); cov = np.linalg.inv(X.T @ (X * w[:, None])); b = cov @ (X.T @ (w * y))
    if weighted is False:   # OLS errors from the known sems
        W = 1 / s ** 2; cov = np.linalg.inv(X.T @ X) @ (X.T * (s ** 2)) @ X @ np.linalg.inv(X.T @ X)
    bgs = np.polyfit(x, gs, 1)[0]
    if mode == "D": return b[1] / bgs, np.sqrt(cov[1, 1]) / bgs
    # h-slope -> ratio: r = <h> + (D_gs/b_gs) dh/dlnL
    hbar = np.average(y, weights=1 / s ** 2); return hbar + (gs.mean() / bgs) * b[1], (gs.mean() / bgs) * np.sqrt(cov[1, 1])
def pooled(rs):
    r = np.array([a for a, b in rs]); e = np.array([b for a, b in rs]); m, se = c.wmean(r, e); return m, se
print("slope ratio r = (dD_bar/dlnL)/(dD_gs/dlnL over same L).  r=1: coefficient unchanged (additive deficit).")
for mode in ("D", "h"):
    for (Lmin, Lmax) in ((24, 99), (18, 99), (18, 30), (12, 99), (30, 99)):
        row = []; rs = []
        for b in (0.5, 1.0, 1.5, 2.0):
            ps = sel(b, Lmin, Lmax)
            if len(ps) < 2: row.append("  --  "); continue
            r, e = ratio(ps, True, mode); row.append("%.2f+-%.2f" % (r, e)); rs.append((r, e)) if b <= 1.5 else None
        m, se = pooled(rs)
        print("  mode=%s L in [%d,%s]: beta=0.5,1,1.5,2: %s | pooled(b<=1.5) = %.2f +- %.2f  [dist from 1: %.1f sig; from 0.5: %.1f sig]" % (
            mode, Lmin, Lmax if Lmax < 99 else "36", "  ".join(row), m, se, (1 - m) / se, (m - 0.5) / se))
# bootstrap of pooled ratio (L>=24) and the result with the number of samples quadrupled (projected precision)
def boot(Lmin, nb=1500):
    out = []
    for _ in range(nb):
        rs = []
        for b in (0.5, 1.0, 1.5):
            ps = sel(b, Lmin); x = np.log([p["L"] for p in ps]); gs = np.array([p["Dgs"] for p in ps])
            y = np.array([p["cols"]["D"][rng.integers(0, p["n"], p["n"])].mean() for p in ps]); s = np.array([p["sem"] for p in ps])
            w = 1 / s ** 2; X = np.column_stack([np.ones_like(x), x]); cov = np.linalg.inv(X.T @ (X * w[:, None])); bb = cov @ (X.T @ (w * y))
            bgs = np.polyfit(x, gs, 1)[0]; rs.append((bb[1] / bgs, np.sqrt(cov[1, 1]) / bgs))
        out.append(pooled(rs)[0])
    return np.mean(out), np.std(out), np.percentile(out, [2.5, 97.5])
for Lmin in (24, 18):
    m, s, ci = boot(Lmin); print("  bootstrap pooled r (beta<=1.5, L>=%d): %.2f +- %.2f, 95%% CI [%.2f, %.2f]" % (Lmin, m, s, ci[0], ci[1]))
# additive vs multiplicative vs ratio=0.6 at beta<=1.5, L>=24 : chi2 of the h data
print("\nchi2 of fixed-beta D_bar(L>=24) (beta=0.5,1,1.5 jointly, 3 free offsets) under models for the L-dependence:")
for name, fn in (("additive (slope = b_gs)", lambda L, gs: gs), ("multiplicative (D=k*Dgs)", None), ("slope = 0.6 b_gs", lambda L, gs: 0.6 * (gs - gs.mean()) + 0), ("slope = 0.3 b_gs", lambda L, gs: 0.3 * (gs - gs.mean())), ("slope = 0 (flat)", lambda L, gs: 0 * gs)):
    tot = 0.0; k = 0
    for b in (0.5, 1.0, 1.5):
        ps = sel(b, 24); y = np.array([p["mean"] for p in ps]); s = np.array([p["sem"] for p in ps]); gs = np.array([p["Dgs"] for p in ps]); L = np.array([p["L"] for p in ps])
        if fn is None:
            kk = np.sum(y * gs / s ** 2) / np.sum(gs ** 2 / s ** 2); m = kk * gs
        else:
            f = fn(L, gs); off = np.sum((y - f) / s ** 2) / np.sum(1 / s ** 2); m = f + off
        tot += float((((y - m) / s) ** 2).sum()); k += len(ps) - 1
    print("   %-26s chi2 = %.2f / %d dof" % (name, tot, k))
