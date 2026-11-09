"""Even L: does the first-order (Thouless) expansion break down, and why?

Compare, for a range of mu (zero-mode splitting), the exact <S2>/q with the
det^2-weighted Thouless value.  A large gap means outcomes with near-singular
Q_S carry O(q)*O(1) weight: the t-expansion fails there.
"""

import numpy as np
from itertools import combinations
import onedim_born as ob

for L in (4, 5, 6, 7, 8):
    Q0, ev0 = ob.chain_state(L)
    N, a = Q0.shape
    reg = ob.half_ring_region(N)
    for mu in (1e-8, 1e-2, 0.3):
        Q, ev = ob.chain_state(L, mu=mu)
        gap = ev[a] - ev[a - 1]
        beta = 4.0
        q = np.exp(-4 * beta)
        outcomes = list(combinations(range(N), a))
        logw = np.array([ob.log_born_weight(Q, E, beta) for E in outcomes])
        p = np.exp(logw - ob.log_normalizer(a, beta))
        d2 = np.empty(len(outcomes))
        th = np.empty(len(outcomes))
        inR = np.zeros(N, dtype=bool)
        inR[reg] = True
        for i, E in enumerate(outcomes):
            E = np.array(E)
            S = np.setdiff1d(np.arange(N), E)
            d = np.linalg.det(Q[S, :])
            d2[i] = d * d
            if abs(d) > 1e-12 * np.sqrt(len(outcomes)):
                X = Q[E, :] @ np.linalg.inv(Q[S, :])
                sel = inR[E][:, None] != inR[S][None, :]
                th[i] = 2.0 * np.sum(X[sel] ** 2)
            else:
                th[i] = np.nan
        s2 = np.array([ob.entropy_trajectory(Q, E, beta, {"h": reg})["h"] for E in outcomes])
        exact = p @ s2 / q / a**2
        thou = np.nansum(d2 * th) / a**2
        frac_bad = np.nansum(d2[~np.isfinite(th)])
        print(f"L={L} mu={mu:.0e} gap={gap:.1e}  exact< S2>/z^2={exact:.6f}  "
              f"thouless={thou:.6f}  diff={exact-thou:+.6f}  det^2 in nodes={frac_bad:.2e}")
    print()
