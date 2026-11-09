import sys, math
sys.dont_write_bytecode = True
sys.path.insert(0, 'D:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030')
import numpy as np
import annealed_gaussian as ag
import crossover_structure as C

pts, base = C.load_points()
print('=== annealed Gaussian vs Born, beta=3 ===')
for p in sorted([q for q in pts if abs(q['beta'] - 3.0) < 0.01], key=lambda q: q['L']):
    L = p['L']
    w = L // 3
    vol = 2 * w * (L - 2 * w) * math.log(2)
    line = 'L=%d beta=%.4f Dbar=%.5f vol=%.3f ratio=%.0f log10=%.2f' % (L, p['beta'], p['mean'], vol, vol / p['mean'], math.log10(vol / p['mean']))
    if L <= 18:
        kap = ag.kappa_finite(L * L, p['beta'])
        line += ' kappa=%.5f annealedD=%.4f' % (kap, ag.annealed_corner(L, kap, mu=1e-8)['D'])
    print(line)
print('=== first-moment diagonal ===')
for L in (6, 12):
    G0 = ag.ground_projector(L, mu=1e-8)
    kap = ag.kappa_finite(L * L, 1.0)
    Gb = ag.annealed_projector(G0, kap)
    print('L=%d kappa=%.4f max|diag(Gbar_def)-diag(G0)|=%.5f, (1-kappa)/L^2=%.5f' % (L, kap, np.abs(np.diag(Gb) - np.diag(G0)).max(), (1 - kap) / L ** 2))
print('=== ground-state slope dD/dlnL ===')
for Ls in ((24, 30, 36), (18, 24, 30, 36)):
    print(Ls, np.polyfit(np.log(Ls), [base[str(L)]['D'] for L in Ls], 1)[0])
