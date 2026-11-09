"""Large-L fixed-beta scan on the weak side (z >> 1), to pin the ln L coefficient.

The only clean weak-measurement points are beta <= 0.5 (z = 3..38 at L = 9..63).
Here we extend to L = 127 to make the coefficient claim a real asymptotic test.
"""

import json
import sys
from multiprocessing import Pool

import numpy as np

SIZES = [45, 63, 91, 127]
BETAS = [0.25, 0.5]


def run_point(job):
    import onedim_born as ob

    L, beta, nsample, seed = job
    rng = np.random.default_rng(seed)
    Q, ev = ob.chain_state(L)
    N, a = Q.shape
    reg = np.arange(N // 2)
    jw = np.exp(ob.log_jweights(a, beta) - ob.log_normalizer(a, beta))
    jc = np.cumsum(jw)
    v = np.empty(nsample)
    for s in range(nsample):
        E = ob.sample_outcome(rng, Q, beta, jw, jc)
        v[s] = ob.renyi2(ob.post_basis(Q, E, beta), reg)
    return {"L": L, "beta": beta, "z": a * np.exp(-2 * beta), "nsample": nsample,
            "S2": float(v.mean()), "sem": float(v.std(ddof=1) / np.sqrt(nsample)),
            "S2_gs": ob.ground_state_entropy(Q, reg)}


if __name__ == "__main__":
    nsample = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    jobs = [(L, b, nsample, 9000 * (i + 1) + L) for i, b in enumerate(BETAS) for L in SIZES]
    with Pool(len(jobs)) as pool:
        rows = list(pool.imap_unordered(run_point, jobs))
    rows.sort(key=lambda r: (r["beta"], r["L"]))
    for r in rows:
        print(f"beta={r['beta']:4.2f} L={r['L']:4d} z={r['z']:8.2f} "
              f"S2={r['S2']:.5f}+-{r['sem']:.5f} gs={r['S2_gs']:.5f} "
              f"ratio={r['S2']/r['S2_gs']:.4f}", flush=True)
    with open("data_large_L.json", "w") as fh:
        json.dump(rows, fh, indent=1)
