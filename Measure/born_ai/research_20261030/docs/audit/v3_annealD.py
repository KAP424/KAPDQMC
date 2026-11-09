import sys
sys.dont_write_bytecode = True
sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030/docs/audit")
from v3_common import *
from math import comb
exec(open(AUD + "/v3_AL.py").read().split("out = {}")[0].split("from v3_common import *")[1])   # ground_G, masks, renyi2
pts = load_all()
def kappa_a(a, beta):
    # high precision via python fractions-free: use decimal
    from decimal import Decimal, getcontext
    getcontext().prec = 60
    t = (Decimal(-2) * Decimal(repr(float(beta)))).exp()
    q = t * t
    def Z(aa):
        s = Decimal(0); c = Decimal(1); qq = Decimal(1)
        for j in range(0, min(aa, 80) + 1):
            if j > 0:
                c = c * (aa - j + 1) / j; qq = qq * q
            s += c * c * qq
        return s
    k = 1 - (1 - t) ** 2 * Z(a - 1) / Z(a)
    return float(k)
G0s = {}
res = []
for (L, b), e in sorted(pts.items()):
    if L not in G0s:
        G0s[L] = ground_G(L)[0]
    G0 = G0s[L]
    kap = kappa_a(L * L, b)
    n = G0.shape[0]
    Gb = 0.5 * np.eye(n) + kap * (G0 - 0.5 * np.eye(n))
    m = masks(L)
    S = {k: renyi2(Gb[np.ix_(m[k], m[k])]) for k in "ABC"}
    Dann = (2 * S["A"] + S["C"]) / 3 - S["B"]
    res.append((L, b, e["z"][-1], kap, Dann, e["mean"], Dann / e["mean"]))
res.sort(key=lambda r: r[6])
print("ratio D_annealedGaussian / D_bar_Born for all 48 points (sorted):")
for r in res[:6]: print("  L=%d beta=%.4f z=%.3f kappa=%.4g D_ann=%.4f D_bar=%.5f ratio=%.2f" % r)
print("  ...")
for r in res[-6:]: print("  L=%d beta=%.4f z=%.3f kappa=%.4g D_ann=%.4f D_bar=%.5f ratio=%.2f" % r)
print("min ratio %.2f ; max ratio %.2f" % (res[0][6], res[-1][6]))
import json
json.dump(res, open(AUD + "/v3_annealD.json", "w"))
