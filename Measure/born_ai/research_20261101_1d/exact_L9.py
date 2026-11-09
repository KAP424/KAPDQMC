"""Exact enumeration for L = 9 (C(18,9) = 48620 outputs): validates the MC and
locates the small-z limit without sampling error."""

import json
import numpy as np
from itertools import combinations
import onedim_born as ob

L = 9
Q, ev = ob.chain_state(L)
N, a = Q.shape
print(f"L={L} N={N} a={a} gap={ev[a]-ev[a-1]:.3e}")
regs = {"half": ob.half_ring_region(N), "third": ob.fraction_region(N, 1.0 / 3)}
G = Q @ Q.T
for k, reg in regs.items():
    print(f"  {k}: f={len(reg)/N:.4f} gs_S2={ob.ground_state_entropy(Q,reg):.6f} "
          f"<N_R>={np.trace(G[np.ix_(reg,reg)]):.6f} V0={ob.ground_state_number_variance(Q,reg):.6f}")

outcomes = np.array(list(combinations(range(N), a)), dtype=np.int64)
res = []
for beta in (0.5, 1.0, 1.5, 2.0, 2.25, 3.0, 4.0):
    logw = np.array([ob.log_born_weight(Q, E, beta) for E in outcomes])
    logp = logw - ob.log_normalizer(a, beta)
    p = np.exp(logp)
    row = {"beta": beta, "z": a * np.exp(-2 * beta), "sum_p": float(p.sum())}
    for k, reg in regs.items():
        s2 = np.empty(len(outcomes))
        for i, E in enumerate(outcomes):
            s2[i] = ob.entropy_trajectory(Q, E, beta, {k: reg})[k]
        row[f"S2_{k}"] = float(p @ s2)
        row[f"S2_{k}_gs"] = ob.ground_state_entropy(Q, reg)
        row[f"S2_{k}_z2"] = float(p @ s2) / row["z"] ** 2
    res.append(row)
    print(f"beta={beta:5.2f} z={row['z']:7.4f} sum_p={row['sum_p']:.10f} "
          f"S2half={row['S2_half']:.6f} /z^2={row['S2_half_z2']:.5f} "
          f"S2third={row['S2_third']:.6f} /z^2={row['S2_third_z2']:.5f}", flush=True)

with open("data_exact_L9.json", "w") as fh:
    json.dump(res, fh, indent=1)
