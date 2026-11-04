"""Referee (I)-a: statistical power at matched z. Models for D(L) at fixed z."""
import sys; sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030/docs/audit")
import numpy as np, ref2_common as c
pts, base = c.load()
groups = []
for p in sorted(pts.values(), key=lambda p: p["z"]):
    for g in groups:
        if abs(g[0]["z"] - p["z"]) / p["z"] < 3e-3:
            g.append(p); break
    else:
        groups.append([p])
groups = [sorted(g, key=lambda q: q["L"]) for g in groups if len(g) >= 3 and g[0]["z"] < 20]

def chi2_fit(y, s, X):
    """weighted LS: y = X b; returns chi2, b, cov"""
    W = 1 / s ** 2
    A = X.T @ (X * W[:, None]); cov = np.linalg.inv(A)
    b = cov @ (X.T @ (W * y)); r = (y - X @ b) / s
    return float((r ** 2).sum()), b, cov

out = []
print("%-7s %-18s | chi2(dof): const | h=const (D~Dgs) | D=Dgs-Delta | D=a+b lnL (b, 1sig) | b/b_gs" % ("z", "L"))
tot = dict(const=0, prop=0, shift=0, slope=0, dof_c=0, dof_s=0)
for g in groups:
    L = np.array([q["L"] for q in g], float); y = np.array([q["mean"] for q in g]); s = np.array([q["sem"] for q in g])
    gs = np.array([q["Dgs"] for q in g]); lnL = np.log(L)
    c0, b0, _ = chi2_fit(y, s, np.ones((len(L), 1)))
    c1, b1, _ = chi2_fit(y, s, gs[:, None])                # D = k * D_gs
    c2, b2, _ = chi2_fit(y - gs, s, np.ones((len(L), 1)))   # D = D_gs + const
    c3, b3, cv3 = chi2_fit(y, s, np.column_stack([np.ones(len(L)), lnL]))
    bgs = np.polyfit(lnL, gs, 1)[0]
    print("%-7.3f %-18s | %5.2f(%d) | %6.2f(%d) | %6.2f(%d) | %+.4f +- %.4f | %+.2f +- %.2f" % (
        g[0]["z"], ",".join(str(int(l)) for l in L), c0, len(L) - 1, c1, len(L) - 1, c2, len(L) - 1,
        b3[1], np.sqrt(cv3[1, 1]), b3[1] / bgs, np.sqrt(cv3[1, 1]) / bgs))
    tot["const"] += c0; tot["prop"] += c1; tot["shift"] += c2; tot["slope"] += c3; tot["dof_c"] += len(L) - 1; tot["dof_s"] += len(L) - 2
print("TOTAL over rows: const %.2f / %d dof; prop-to-Dgs %.2f / %d; Dgs-shift %.2f / %d; free slope %.2f / %d" % (
    tot["const"], tot["dof_c"], tot["prop"], tot["dof_c"], tot["shift"], tot["dof_c"], tot["slope"], tot["dof_s"]))
# common multiplicative drift gamma: D(L,z)=f(z)*(1+gamma*ln(L/18)) over ALL rows with >=3 L (incl z<=3.2 and 5.93,10.55)
print("\nCommon drift test D = f_row * (1 + g ln(L/18)), g common to all rows (profile over f_row):")
rows = [(np.array([q["L"] for q in g], float), np.array([q["mean"] for q in g]), np.array([q["sem"] for q in g])) for g in groups]
def prof(gam):
    t = 0.0
    for L, y, s in rows:
        m = 1 + gam * np.log(L / 18.0); w = 1 / s ** 2
        f = (w * m * y).sum() / (w * m * m).sum(); t += (w * (y - f * m) ** 2).sum()
    return t
gs_ = np.linspace(-0.4, 0.4, 801); ch = np.array([prof(g) for g in gs_]); i = ch.argmin()
band = gs_[ch < ch[i] + 1]; band4 = gs_[ch < ch[i] + 3.84]
print("  best g = %+.3f (chi2 %.2f vs %.2f at g=0; dof %d); 1sig [%.3f,%.3f], 95%% [%.3f,%.3f]" % (
    gs_[i], ch[i], prof(0.0), tot["dof_c"] - len(rows), band.min(), band.max(), band4.min(), band4.max()))
print("  => over L=18->36 (ln2=0.693) D at fixed z may change by %+.1f%% .. %+.1f%% (95%%)" % (100 * band4.min() * np.log(2), 100 * band4.max() * np.log(2)))
# what 'D ~ Dgs' would imply: relative change of D_gs between L=18 and 36
print("  D_gs(36)/D_gs(18) - 1 = %.1f%%" % (100 * (base['36']['D'] / base['18']['D'] - 1)))
