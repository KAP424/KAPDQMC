"""Parallel production scan (multiprocessing over (L, beta) points).

Usage:
    python run_onedim_parallel.py matched 8192 12
    python run_onedim_parallel.py fixed   8192 12

z = a t = L exp(-2 beta); a = L is the particle number, N = 2L the site number.
"""

import json
import os
import sys
from multiprocessing import Pool

import numpy as np

SIZES = [9, 15, 21, 27, 33, 45, 63]
Z_VALUES = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
BETAS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]


def regions_for(N):
    regs = {"half": np.arange(N // 2)}
    if N % 6 == 0:
        regs["third"] = np.arange(N // 3)
    return regs


def run_point(job):
    import onedim_born as ob

    L, beta, nsample, seed = job
    rng = np.random.default_rng(seed)
    Q, ev = ob.chain_state(L)
    N, a = Q.shape
    regs = regions_for(N)
    jw = np.exp(ob.log_jweights(a, beta) - ob.log_normalizer(a, beta))
    jc = np.cumsum(jw)
    acc = {k: np.empty(nsample) for k in regs}
    for s in range(nsample):
        E = ob.sample_outcome(rng, Q, beta, jw, jc)
        vals = ob.entropy_trajectory(Q, E, beta, regs)
        for k in regs:
            acc[k][s] = vals[k]
    out = {"L": L, "beta": beta, "N": N, "a": a, "nsample": nsample,
           "z": a * np.exp(-2 * beta), "gap": float(ev[a] - ev[a - 1])}
    for k, reg in regs.items():
        v = acc[k]
        out[f"S2_{k}"] = float(v.mean())
        out[f"S2_{k}_sem"] = float(v.std(ddof=1) / np.sqrt(nsample))
        out[f"S2_{k}_gs"] = ob.ground_state_entropy(Q, reg)
        out[f"V0_{k}"] = ob.ground_state_number_variance(Q, reg)
        out[f"f_{k}"] = len(reg) / N
        out[f"pred_{k}"] = (N * len(reg) - 4 * len(reg) * np.trace(
            (Q @ Q.T)[np.ix_(reg, reg)]) + 4 * np.trace(
            (Q @ Q.T)[np.ix_(reg, reg)]) ** 2
            + 4 * ob.ground_state_number_variance(Q, reg)) / a**2
    return out


def main():
    mode = sys.argv[1]
    nsample = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
    nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    jobs = []
    if mode == "matched":
        for zi, z in enumerate(Z_VALUES):
            for L in SIZES:
                beta = 0.5 * np.log(L / z)
                if beta <= 0:
                    continue
                jobs.append((L, beta, nsample, 1000 * zi + L))
    elif mode == "fixed":
        for bi, beta in enumerate(BETAS):
            for L in SIZES:
                jobs.append((L, beta, nsample, 7000 * (bi + 1) + L))
    else:
        raise SystemExit("mode must be matched|fixed")

    with Pool(min(nproc, len(jobs))) as pool:
        rows = []
        for r in pool.imap_unordered(run_point, jobs):
            rows.append(r)
            print(f"L={r['L']:3d} beta={r['beta']:6.3f} z={r['z']:8.3f} "
                  f"S2half={r['S2_half']:.5f}+-{r['S2_half_sem']:.5f} "
                  f"gs={r['S2_half_gs']:.5f}", flush=True)
    rows.sort(key=lambda r: (round(np.log(r["z"]), 6), r["L"]))
    with open(f"data_{mode}.json", "w") as fh:
        json.dump(rows, fh, indent=1)
    print("wrote", f"data_{mode}.json", len(rows), "points")


if __name__ == "__main__":
    main()
