import sys, glob, os
sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030/docs/audit")
from v3_common import *
pts = load_all(); base = baseline(); Dgs = {int(k): v["D"] for k, v in base.items()}
# per-sample time from elapsed column
tt = {}
for d in DIRS:
    for f in sorted(glob.glob(f"{R}/{d}/*.csv")):
        hdr, a = read_csv(f)
        if "elapsed" in hdr:
            L = int(a[0, 0]); el = a[:, hdr.index("elapsed")]
            tt.setdefault(L, []).append(el[-1] / a.shape[0])
print("per-sample seconds (from elapsed column) by L:", {L: (round(float(np.mean(v)), 2), len(v)) for L, v in sorted(tt.items())})
# slope-ratio error projection for beta=1.5 and 2
def slope_err(sig, Ls):
    x = np.log(Ls); w = 1 / np.asarray(sig)**2
    xb = (w * x).sum() / w.sum()
    return 1 / np.sqrt((w * (x - xb)**2).sum())
g = np.polyfit(np.log([24, 30, 36]), [Dgs[24], Dgs[30], Dgs[36]], 1)[0]
print("gs slope", g)
for b in (1.5, 2.0):
    sel = sorted([(L, e) for (L, bb), e in pts.items() if abs(bb - b) < 2e-3 and L >= 24])
    sig = [e["sem"] for L, e in sel]; ns = [e["n"] for L, e in sel]; Ls = [L for L, e in sel]
    now = slope_err(sig, Ls) / g
    target_n = {24: 8192, 30: 4096, 36: 2048}
    sig2 = [s * np.sqrt(n / target_n[L]) for s, n, L in zip(sig, ns, Ls)]
    after = slope_err(sig2, Ls) / g
    # with L=42 at 2048 samples: sem scaling ~ sigma(L) from L=36: assume per-sample sd ~ grows ~ L (approx from data) -> use sd36 * (42/36)
    sd36 = sig[2] * np.sqrt(ns[2]); sd42 = sd36 * 42 / 36
    sig3 = sig2 + [sd42 / np.sqrt(2048)]; L3 = Ls + [42]
    after3 = slope_err(sig3, L3) / g
    print("beta=%.1f: slope-ratio error now %.3f ; after sample increase %.3f ; plus L=42(2048) %.3f   [n now %s]" % (b, now, after, after3, ns))
# combine two betas
def comb(errs): return 1 / np.sqrt(sum(1 / e**2 for e in errs))
# CPU hours
t = {L: float(np.mean(v)) for L, v in tt.items()}
print("t:", t)
