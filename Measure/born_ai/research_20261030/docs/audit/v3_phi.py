import sys
sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030/docs/audit")
from v3_common import *
pts = load_all()
print("phi(z) = (Dbar/z^2)/(16/81), inverse-variance pooled over L")
for zt in (0.089, 0.357, 0.803, 1.428, 3.213, 5.935, 10.55):
    ms, es = [], []
    for (L, b), e in pts.items():
        z = e["z"][-1]
        if abs(z - zt) / zt < 3e-3:
            ms.append(e["mean"] / z**2); es.append(e["sem"] / z**2)
    ms, es = np.array(ms), np.array(es); w = 1 / es**2
    mu, se = (w * ms).sum() / w.sum(), 1 / np.sqrt(w.sum())
    print("  z=%.3f : D/z^2 = %.4f +- %.4f ; phi = %.3f +- %.3f  (n_L=%d)" % (zt, mu, se, mu / (16 / 81), se / (16 / 81), len(ms)))
