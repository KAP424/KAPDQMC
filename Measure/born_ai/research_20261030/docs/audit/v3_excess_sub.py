import sys
sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030/docs/audit")
from v3_common import *
pts = load_all(); base = baseline(); Dgs = {int(k): v["D"] for k, v in base.items()}
def stat(sel, label):
    d = np.array([e["mean"] - Dgs[L] for (L, b), e in sel]); s = np.array([e["sem"] for k, e in sel]); w = 1 / s**2
    mu = (w * d).sum() / w.sum(); se = 1 / np.sqrt(w.sum()); chi = (w * (d - mu)**2).sum()
    print("%-34s n=%d  excess %+.4f +- %.4f (%.2f sigma)  chi2/dof %.2f/%d" % (label, len(d), mu, se, mu / se, chi, len(d) - 1))
all_sel = [((L, b), e) for (L, b), e in sorted(pts.items()) if b <= 1.0 + 1e-9 and L >= 12]
stat(all_sel, "beta<=1, L=12..36")
stat([x for x in all_sel if x[0][0] >= 24], "beta<=1, L>=24")
stat([x for x in all_sel if x[0][0] >= 18], "beta<=1, L>=18")
stat([x for x in all_sel if x[0][0] >= 30], "beta<=1, L>=30")
stat([x for x in all_sel if x[0][0] <= 24], "beta<=1, L<=24")
stat([x for x in all_sel if abs(x[0][1] - 0.5) < 2e-3 or x[0][0] == 12], "beta=0.5 (+L=12 z=43.85)")
stat([x for x in all_sel if abs(x[0][1] - 1.0) < 2e-3], "beta=1.0 only")
# including z>=40 with beta<=1.29 (the L=24 beta=1.2877)
sel2 = [((L, b), e) for (L, b), e in sorted(pts.items()) if e["z"][-1] >= 40 and b <= 1.3 and L >= 12]
stat(sel2, "z>=40, beta<=1.29, L>=12")
# relative excess variants
sel = all_sel
h1 = np.array([e["mean"] / Dgs[L] - 1 for (L, b), e in sel]); sh = np.array([e["sem"] / Dgs[L] for (L, b), e in sel]); w = 1 / sh**2
print("relative excess (weighted mean of h-1): %+.4f +- %.4f ; unweighted %+.4f" % ((w * h1).sum() / w.sum(), 1 / np.sqrt(w.sum()), h1.mean()))
d = np.array([e["mean"] - Dgs[L] for (L, b), e in sel]); s = np.array([e["sem"] for k, e in sel]); w2 = 1 / s**2
print("abs excess / weighted-mean D_gs = %.4f ; D_gs weighted mean = %.4f" % ((w2 * d).sum() / w2.sum() / ((w2 * np.array([Dgs[L] for (L, b), e in sel])).sum() / w2.sum()), (w2 * np.array([Dgs[L] for (L, b), e in sel])).sum() / w2.sum()))
# beta=1.5 relative
sel15 = [((L, b), e) for (L, b), e in sorted(pts.items()) if abs(b - 1.5) < 2e-3 and L >= 24]
h = np.array([e["mean"] / Dgs[L] - 1 for (L, b), e in sel15]); sh = np.array([e["sem"] / Dgs[L] for (L, b), e in sel15]); w = 1 / sh**2
print("beta=1.5 L>=24 relative: %+.4f +- %.4f" % ((w * h).sum() / w.sum(), 1 / np.sqrt(w.sum())))
# all z>=40 h range
allz = [(e["mean"] / Dgs[L], L, b) for (L, b), e in pts.items() if e["z"][-1] >= 40]
print("z>=40 pooled h range: %.4f .. %.4f" % (min(a[0] for a in allz), max(a[0] for a in allz)), "; n =", len(allz))
