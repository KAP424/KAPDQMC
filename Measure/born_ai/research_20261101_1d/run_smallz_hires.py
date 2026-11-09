"""Extra statistics on the two smallest-z rows only.

z = 0.03 and z = 0.1 are the rows that actually test the small-z theorem, and
they are also the rows with the largest error bars in panel (a) of the summary
figure (2-6%, because S2 ~ z^2 is tiny there and the estimator is noisy).

Everything else in the matched scan already has sub-percent errors, so there is
no point re-running it.  This script does only those two rows at 4x the sample
count of run_onedim_big.py and merges the result into data_matched_big.json.

Usage:  python run_smallz_hires.py 131072 2
"""

import json
import os
import sys
from multiprocessing import Pool

import numpy as np

SIZES = [9, 15, 21, 27, 33, 45, 63]
Z_VALUES = [0.03, 0.1]


def run_point(job):
    import onedim_born as ob

    L, beta, nsample, seed = job
    rng = np.random.default_rng(seed)
    Q, ev = ob.chain_state(L)
    N, n = Q.shape
    regs = {"half": np.arange(N // 2)}
    jw = np.exp(ob.log_jweights(n, beta) - ob.log_normalizer(n, beta))
    jc = np.cumsum(jw)
    acc = np.empty(nsample)
    for s in range(nsample):
        E = ob.sample_outcome(rng, Q, beta, jw, jc)
        acc[s] = ob.entropy_trajectory(Q, E, beta, regs)["half"]

    out = {"L": L, "beta": beta, "N": N, "a": n, "nsample": nsample,
           "z": n * np.exp(-2 * beta), "gap": float(ev[n] - ev[n - 1])}
    G = Q @ Q.T
    reg = regs["half"]
    nR = float(np.trace(G[np.ix_(reg, reg)]))
    V0 = ob.ground_state_number_variance(Q, reg)
    out["S2_half"] = float(acc.mean())
    out["S2_half_sem"] = float(acc.std(ddof=1) / np.sqrt(nsample))
    out["S2_half_gs"] = ob.ground_state_entropy(Q, reg)
    out["V0_half"] = V0
    out["f_half"] = len(reg) / N
    out["pred_half"] = (N * len(reg) - 4 * len(reg) * nR + 4 * nR**2
                        + 4 * V0) / n**2
    return out


def main():
    nsample = int(sys.argv[1]) if len(sys.argv) > 1 else 131072
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    jobs = []
    for zi, z in enumerate(Z_VALUES):
        for L in SIZES:
            beta = 0.5 * np.log(L / z)
            jobs.append((L, beta, nsample, 50_000_000 + 1000 * zi + L))
    jobs.sort(key=lambda j: -j[0])          # longest (largest L) first

    print(f"{len(jobs)} jobs x {nsample} samples, {min(nproc, len(jobs))} workers",
          flush=True)
    # Each finished point is checkpointed to its own file.  A 14-point run is
    # ~2 CPU-h, and the merge below only happens at the very end, so without
    # this a kill at 13/14 throws away everything.
    ckpt = "smallz_hires_checkpoint.json"
    with Pool(min(nproc, len(jobs))) as pool:
        rows = []
        for r in pool.imap_unordered(run_point, jobs):
            rows.append(r)
            json.dump(rows, open(ckpt, "w"), indent=1)
            print(f"[{len(rows)}/{len(jobs)}] L={r['L']:3d} z={r['z']:6.3f} "
                  f"S2/z2={r['S2_half']/r['z']**2:.5f}"
                  f"+-{r['S2_half_sem']/r['z']**2:.5f}", flush=True)

    path = "data_matched_big.json"
    old = [r for r in json.load(open(path)) if "S2_half" in r]
    new = [r for r in old
           if not any(abs(r["z"] - n["z"]) < 1e-3 and r["L"] == n["L"]
                      for n in rows)]
    merged = new + rows
    merged.sort(key=lambda r: (round(np.log(r["z"]), 6), r["L"]))
    json.dump(merged, open(path, "w"), indent=1)
    os.remove(ckpt)
    print(f"replaced {len(old) - len(new)} points, "
          f"data_matched_big.json now {len(merged)} points")


if __name__ == "__main__":
    main()
