"""Referee (I)-d6: power-law single-scale z* ~ L^p (the retracted z*~L^0.8 family) -- chi2 vs p, with and without L=36."""
import sys; sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030/docs/audit")
import numpy as np
import ref2_shift_lib as lib
import ref2_stats as pv
for (hlo, hhi, Lmin, Lmax, K) in ((0.25, 1.15, 12, 36, 8), (0.25, 1.15, 12, 30, 8), (0.35, 1.15, 12, 36, 8), (0.35, 1.15, 12, 30, 8), (0.25, 1.15, 18, 36, 8)):
    pr = lib.Prob(hlo, hhi, Lmin, K, Lmax); ref = pr.Ls[0]
    ps = np.linspace(0.0, 2.2, 89); ch = []
    for p in ps:
        sh = {l: p * np.log(l / ref) for l in pr.Ls}; ch.append(pr.chi(sh))
    ch = np.array(ch); i = ch.argmin(); dof = pr.N - K - 1
    ok = ps[ch < ch[i] + 1]
    print("h[%.2f,%.2f] L[%d,%d] K=%d N=%d: best p=%.2f (1sig %.2f..%.2f) chi2=%.1f/%d (p-value %.1e) | chi2 at p=0.8: %.1f ; p=1.0: %.1f ; p=1.8: %.1f ; p=0 (pure z): %.1f" % (
        hlo, hhi, Lmin, Lmax, K, pr.N, ps[i], ok.min(), ok.max(), ch[i], dof, pv.pchi(ch[i], dof), ch[np.argmin(abs(ps - 0.8))], ch[np.argmin(abs(ps - 1.0))], ch[np.argmin(abs(ps - 1.8))], ch[0]))
