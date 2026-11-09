import sys
sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030/docs/audit")
from v3_common import *
pts = load_all()
for zt in (0.089, 0.357):
    ms, es = [], []
    for (L, b), e in pts.items():
        z = e["z"][-1]
        if abs(z - zt) / zt < 3e-3 and L <= 30:
            x = e["DV"] / z**2
            ms.append(x.mean()); es.append(x.std(ddof=1) / np.sqrt(x.size))
    ms, es = np.array(ms), np.array(es); w = 1 / es**2
    mu, se = (w * ms).sum() / w.sum(), 1 / np.sqrt(w.sum())
    print("z=%.3f: D_V/z^2 pooled = %.4f +- %.4f ; 8/81 = %.4f ; ratio %.3f +- %.3f" % (zt, mu, se, 8/81, mu/(8/81), se/(8/81)))
# beta0 DV vs D_V0
AL = json.load(open(AUD + "/v3_AL.json"))
for L in (6, 12, 18, 24, 30, 36):
    hdr, a = read_csv(f"{R}/data_beta0/L{L}_beta0.0000.csv")
    print("L=%d beta0 DV = %.8f ; D_V0(mine) = %.8f ; diff %.1e" % (L, a[:, hdr.index('DV')].mean(), AL[str(L)]["DV0"], a[:, hdr.index('DV')].mean() - AL[str(L)]["DV0"]))
