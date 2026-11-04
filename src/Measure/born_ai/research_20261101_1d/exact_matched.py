"""Exact fixed-z values at small L, to test whether the L-drift is real.

For each (L, z) with beta = ln(L/z)/2 we enumerate all fixed-half outcomes.
"""

import json
import numpy as np
from itertools import combinations
import onedim_born as ob

Z_VALUES = [0.3, 1.0, 3.0]
out = []
for L in (5, 7, 9):
    Q, ev = ob.chain_state(L)
    N, a = Q.shape
    reg = ob.half_ring_region(N)
    outcomes = np.array(list(combinations(range(N), a)), dtype=np.int64)
    for z in Z_VALUES:
        beta = 0.5 * np.log(L / z)
        logw = np.array([ob.log_born_weight(Q, E, beta) for E in outcomes])
        p = np.exp(logw - ob.log_normalizer(a, beta))
        s2 = np.array([ob.entropy_trajectory(Q, E, beta, {"h": reg})["h"] for E in outcomes])
        row = {"L": L, "beta": beta, "z": a * np.exp(-2 * beta),
               "S2": float(p @ s2), "sum_p": float(p.sum()),
               "S2_gs": ob.ground_state_entropy(Q, reg), "nout": len(outcomes)}
        out.append(row)
        print(f"L={L} z={z:5.2f} beta={beta:6.3f} S2={row['S2']:.6f} "
              f"z^2={row['z']**2:.5f} S2/z^2={row['S2']/row['z']**2:.5f} "
              f"sum_p={row['sum_p']:.10f}", flush=True)

with open("data_exact_matched.json", "w") as fh:
    json.dump(out, fh, indent=1)
