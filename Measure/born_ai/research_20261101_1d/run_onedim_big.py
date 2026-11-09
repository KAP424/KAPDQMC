"""Higher-statistics 1D scan: 4x the samples of run_onedim_parallel.py.

The 1D cost is tiny (a few ms per sample up to L=63), so the whole grid can be
done at 32768 samples per point instead of 4096/8192.  This removes the visible
error bars in the published figures and, more importantly, makes the fixed-z
chi^2 test meaningful at the smallest z where the theorem is being checked.

Outputs data_matched_big.json / data_fixed_big.json with the same schema as the
original files, so analyze_onedim.py / make_figures.py can consume either.

Usage:  python run_onedim_big.py matched 32768 13
        python run_onedim_big.py fixed   32768 13
"""

import json
import sys
from multiprocessing import Pool

import numpy as np

SIZES = [9, 15, 21, 27, 33, 45, 63]
Z_VALUES = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
BETAS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]


def run_point(job):
    import onedim_born as ob

    L, beta, nsample, seed = job
    rng = np.random.default_rng(seed)
    Q, ev = ob.chain_state(L)
    N, n = Q.shape
    regs = {"half": np.arange(N // 2)}
    if N % 6 == 0:
        regs["third"] = np.arange(N // 3)
    jw = np.exp(ob.log_jweights(n, beta) - ob.log_normalizer(n, beta))
    jc = np.cumsum(jw)
    acc = {k: np.empty(nsample) for k in regs}
    for s in range(nsample):
        E = ob.sample_outcome(rng, Q, beta, jw, jc)
        vals = ob.entropy_trajectory(Q, E, beta, regs)
        for k in regs:
            acc[k][s] = vals[k]
    out = {"L": L, "beta": beta, "N": N, "a": n, "nsample": nsample,
           "z": n * np.exp(-2 * beta), "gap": float(ev[n] - ev[n - 1])}
    G = Q @ Q.T
    for k, reg in regs.items():
        v = acc[k]
        nR = float(np.trace(G[np.ix_(reg, reg)]))
        V0 = ob.ground_state_number_variance(Q, reg)
        out[f"S2_{k}"] = float(v.mean())
        out[f"S2_{k}_sem"] = float(v.std(ddof=1) / np.sqrt(nsample))
        out[f"S2_{k}_gs"] = ob.ground_state_entropy(Q, reg)
        out[f"V0_{k}"] = V0
        out[f"f_{k}"] = len(reg) / N
        out[f"pred_{k}"] = (N * len(reg) - 4 * len(reg) * nR + 4 * nR**2
                            + 4 * V0) / n**2
    return out


def main():
    mode = sys.argv[1]
    nsample = int(sys.argv[2]) if len(sys.argv) > 2 else 32768
    nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 13
    jobs = []
    if mode == "matched":
        for zi, z in enumerate(Z_VALUES):
            for L in SIZES:
                beta = 0.5 * np.log(L / z)
                if beta <= 0:
                    continue
                jobs.append((L, beta, nsample, 20_000_000 + 1000 * zi + L))
    elif mode == "fixed":
        for bi, beta in enumerate(BETAS):
            for L in SIZES:
                jobs.append((L, beta, nsample, 30_000_000 + 7000 * (bi + 1) + L))
    else:
        raise SystemExit("mode must be matched|fixed")

    with Pool(min(nproc, len(jobs))) as pool:
        rows = []
        for r in pool.imap_unordered(run_point, jobs):
            rows.append(r)
            print(f"L={r['L']:3d} beta={r['beta']:6.3f} z={r['z']:8.3f} "
                  f"S2half={r['S2_half']:.6f}+-{r['S2_half_sem']:.6f} "
                  f"pred={r['pred_half']:.5f}", flush=True)
    rows.sort(key=lambda r: (round(np.log(r["z"]), 6), r["L"]))
    with open(f"data_{mode}_big.json", "w") as fh:
        json.dump(rows, fh, indent=1)
    print("wrote", f"data_{mode}_big.json", len(rows), "points")


if __name__ == "__main__":
    main()
