"""Referee (I)-b: can the beta=1.5 / beta=2 plateau exclude a slowly rising h?"""
import sys; sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030/docs/audit")
import numpy as np, ref2_common as c
pts, base = c.load()
def fixed(beta, Lmin):
    return sorted([p for p in pts.values() if abs(p["beta"] - beta) < 2e-3 and p["L"] >= Lmin], key=lambda p: p["L"])
def slope_h(ps):
    x = np.log([p["L"] for p in ps]); y = np.array([p["mean"] / p["Dgs"] for p in ps]); s = np.array([p["sem"] / p["Dgs"] for p in ps])
    X = np.column_stack([np.ones_like(x), x]); W = 1 / s ** 2
    cov = np.linalg.inv(X.T @ (X * W[:, None])); b = cov @ (X.T @ (W * y))
    chi = (W * (y - X @ b) ** 2).sum()
    return b[1], np.sqrt(cov[1, 1]), chi, len(y) - 2, x
print("dh/dlnL (h = D_bar/D_gs) at fixed beta; Delta_h(24->36) = slope*ln1.5")
for beta in (0.5, 1.0, 1.5, 2.0):
    for Lmin in (24, 18):
        ps = fixed(beta, Lmin)
        b, e, chi, dof, x = slope_h(ps)
        print("  beta=%.1f L>=%d: slope=%+.3f +- %.3f  -> h(36)-h(24) = %+.3f +- %.3f   [95%% range %+.3f..%+.3f] chi2=%.2f/%d" % (
            beta, Lmin, b, e, b * np.log(1.5), e * np.log(1.5), (b - 1.96 * e) * np.log(1.5), (b + 1.96 * e) * np.log(1.5), chi, dof))
# beta<=1.5 joint (h-1 offsets differ per beta: fit common slope, separate intercepts)
def joint_slope(betas, Lmin):
    rows = []
    for k, b in enumerate(betas):
        for p in fixed(b, Lmin):
            rows.append((k, np.log(p["L"]), p["mean"] / p["Dgs"], p["sem"] / p["Dgs"]))
    k = np.array([r[0] for r in rows]); x = np.array([r[1] for r in rows]); y = np.array([r[2] for r in rows]); s = np.array([r[3] for r in rows])
    X = np.column_stack([np.eye(len(betas))[k], x]); W = 1 / s ** 2
    cov = np.linalg.inv(X.T @ (X * W[:, None])); bb = cov @ (X.T @ (W * y))
    return bb[-1], np.sqrt(cov[-1, -1]), (W * (y - X @ bb) ** 2).sum(), len(y) - X.shape[1]
for betas in ((0.5, 1.0, 1.5), (1.0, 1.5), (0.5, 1.0, 1.5, 2.0)):
    b, e, chi, dof = joint_slope(betas, 24)
    print("  joint beta%s L>=24: common dh/dlnL = %+.3f +- %.3f (chi2 %.2f/%d) -> 36/24 change %+.3f +- %.3f" % (list(betas), b, e, chi, dof, b * np.log(1.5), e * np.log(1.5)))
# What would a 'slowly rising h' look like?  Scenarios fitted to L=24,30,36 at beta=1.5 :
ps = fixed(1.5, 24)
L = np.array([p["L"] for p in ps], float); y = np.array([p["mean"] for p in ps]); s = np.array([p["sem"] for p in ps]); gs = np.array([p["Dgs"] for p in ps])
print("\nbeta=1.5, L=24,30,36: D_bar = %s  (Dgs=%s)" % (np.round(y, 4), np.round(gs, 4)))
def chi(model): return float((((y - model) / s) ** 2).sum())
print("  chi2 constant h (best-fit k*Dgs):  %.2f" % min(chi(k * gs) for k in np.linspace(0.8, 1.1, 3001)))
for target in (0.97, 0.98, 1.0):
    # scenario: h(L) rises linearly in ln L from h(24)=0.948 reaching `target` at L=72 (needs slope)
    for Lhit in (72, 100, 200):
        sl = (target - 0.948) / np.log(Lhit / 24.0)
        hmod = 0.948 + sl * np.log(L / 24.0)
        # allow free offset:
        offs = np.linspace(0.85, 1.05, 4001)
        best = min(((((y - (o + sl * np.log(L / 24.0)) * gs) / s) ** 2).sum(), o) for o in offs)
        print("  h(L)=h24+%.3f lnL/24 (reaches %.2f at L=%d): best chi2 %.2f (offset %.3f)" % (sl, target, Lhit, best[0], best[1]))
# Power for single-scale with p: h = H(z/z*), z* ~ L^p at beta=1.5: expected rise of h from L=24 to 36 using the measured local slope dh/dln z
print("\nLocal slope of h vs ln z at L=24 near beta=1.5 (from measured neighbours):")
pp = sorted([p for p in pts.values() if p["L"] == 24], key=lambda p: p["z"])
for a, b in zip(pp[:-1], pp[1:]):
    if 10 < a["z"] < 100:
        dh = (b["mean"] / b["Dgs"] - a["mean"] / a["Dgs"]); dz = np.log(b["z"] / a["z"])
        print("   z %.1f->%.1f: h %.3f->%.3f slope %.3f per ln z" % (a["z"], b["z"], a["mean"] / a["Dgs"], b["mean"] / b["Dgs"], dh / dz))
print("  Single-scale prediction for h(36)-h(24) at beta=1.5 = slope * (2-p) * ln1.5; z/z* ~ L^(2-p):")
for sl in (0.10, 0.17, 0.25):
    print("    local slope %.2f:  " % sl + "  ".join("p=%.1f: %+.3f" % (pexp, sl * (2 - pexp) * np.log(1.5)) for pexp in (0.6, 0.8, 1.0, 1.3, 1.6, 1.8)))
