"""Reproduce the summary's paired-bootstrap p(0.8)-p(0.5), p(0.5)-p(0.4) on the summary's own (per-threshold) L-sets, and on L>=12."""
import sys; sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030/docs/audit")
import numpy as np
exec(open("ref2_I_contours.py").read().split("def pair(")[0].split("report(None")[0])   # reuse loader, T, boots, pfit
def pairS(h1, h2, Lset=None):
    d0 = pfit(T, h2, Lset)[0] - pfit(T, h1, Lset)[0]; db = []
    for Tb in boots:
        a, b = pfit(Tb, h1, Lset)[0], pfit(Tb, h2, Lset)[0]
        if a is not None and b is not None: db.append(b - a)
    return d0, np.std(db)
for (h1, h2) in ((0.4, 0.5), (0.5, 0.8), (0.3, 0.9), (0.5, 0.9)):
    d0, e = pairS(h1, h2); print("summary-style L-sets: p(%.1f)-p(%.1f) = %+.3f +- %.3f (%.1f sigma)" % (h2, h1, d0, e, d0 / e))
