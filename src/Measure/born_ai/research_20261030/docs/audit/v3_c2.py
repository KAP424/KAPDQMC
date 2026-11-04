import sys
sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030/docs/audit")
from v3_common import *
base = baseline()
Ls = np.array(sorted(int(k) for k in base)); D = np.array([base[str(k)]["D"] for k in Ls])
print("L range in baseline:", Ls.min(), Ls.max(), len(Ls))
models = [("pure log", lambda L: None), ("+a/L", lambda L: 1 / L), ("+a/L^0.5", lambda L: 1 / np.sqrt(L)), ("+a/L^1.5", lambda L: 1 / L**1.5)]
for lo in (24, 6):
    m = Ls >= lo; L = Ls[m].astype(float); d = D[m]
    out = []
    for name, f in models:
        cols = [np.log(L)] + ([f(L)] if f(L) is not None else []) + [np.ones_like(L)]
        X = np.column_stack(cols)
        coef = np.linalg.lstsq(X, d, rcond=None)[0]
        r = d - X @ coef
        out.append("%s: c2=%.4f (rms %.1e)%s" % (name, coef[0], np.sqrt(np.mean(r**2)), "" if f(L) is None else " [a=%.3f, b=%.3f]" % (coef[1], coef[2])))
    print("L>=%d:" % lo, " | ".join(out))
# the L>=24 +a/L fit in detail
m = Ls >= 24; L = Ls[m].astype(float); d = D[m]
X = np.column_stack([np.log(L), 1 / L, np.ones_like(L)]); coef = np.linalg.lstsq(X, d, rcond=None)[0]
r = d - X @ coef
print("L>=24 log + a/L: c2 = %.5f ; 1/L coef = %.4f ; const = %.4f ; rms = %.2e" % (coef[0], coef[1], coef[2], np.sqrt(np.mean(r**2))))
# alternative: rms of sample (ddof) 
print("   rms (ddof=3) = %.2e ; max|r| = %.2e" % (np.sqrt((r**2).sum() / (len(r) - 3)), np.abs(r).max()))
# rank other models at L>=24 
for name, f in models[2:]:
    X = np.column_stack([np.log(L), f(L), np.ones_like(L)]); cf = np.linalg.lstsq(X, d, rcond=None)[0]; rr = d - X @ cf
    print("   ", name, "rms %.2e ; ratio to a/L rms: %.1f" % (np.sqrt(np.mean(rr**2)), np.sqrt(np.mean(rr**2)) / np.sqrt(np.mean(r**2))))
# ground state slope over 24,30,36
print("gs slope over L=24,30,36: %.5f" % np.polyfit(np.log([24, 30, 36]), [base["24"]["D"], base["30"]["D"], base["36"]["D"]], 1)[0])
