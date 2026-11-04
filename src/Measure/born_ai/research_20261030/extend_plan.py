"""Plan and launch the sample extension for the 2D production grid.

Cost model measured on this machine (ms per sample, single BLAS thread):
    L=6  0.1    L=12 3.2    L=18 66    L=24 246    L=30 810    L=36 3912

The worst relative errors are at small z (D is tiny there) and at large L
(512 samples only).  We allocate so that the *absolute* SEM on D targets
0.002 for L <= 24 and 0.006 for L >= 30 (the large-L points are dominated by
L=36 cost), capped by a wall-clock budget.

Usage:
    python extend_plan.py --budget 3600            # dry run, print plan
    python extend_plan.py --budget 3600 --run 4    # launch 4 parallel workers
"""

import argparse
import csv
import glob
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
MS_PER_SAMPLE = {6: 0.1, 12: 3.2, 18: 66.0, 24: 246.0, 30: 810.0, 36: 3912.0}


def read_points():
    pts = []
    for f in sorted(glob.glob(os.path.join(HERE, "data_matched_z", "*.csv"))):
        with open(f) as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            continue
        L = int(float(rows[0]["L"]))
        beta = float(rows[0]["beta"])
        z = float(rows[0]["z"])
        d = [float(r["D"]) for r in rows]
        n = len(d)
        mean = sum(d) / n
        var = sum((x - mean) ** 2 for x in d) / max(n - 1, 1)
        sem = (var / n) ** 0.5
        pts.append(dict(file=f, L=L, beta=beta, z=z, n=n, mean=mean, sem=sem))
    return pts


def plan(pts, budget_s, nproc, target_abs):
    """Return list of (point, nextra) with estimated seconds."""
    # priority: largest relative error first, but weighted by cheapness
    cands = []
    for p in pts:
        L = p["L"]
        tgt = target_abs[0] if L <= 24 else target_abs[1]
        if p["sem"] <= tgt or p["sem"] == 0:
            continue
        need = p["n"] * (p["sem"] / tgt) ** 2 - p["n"]
        need = int(max(0, need))
        # round up to a multiple of 256, and never less than 256
        need = max(256, ((need + 255) // 256) * 256)
        cost = need * MS_PER_SAMPLE.get(L, 4000.0) / 1000.0
        cands.append((p, need, cost))
    cands.sort(key=lambda c: -(c[0]["sem"] / (target_abs[0] if c[0]["L"] <= 24
                                              else target_abs[1])))
    total_wall = budget_s * nproc
    chosen, spent = [], 0.0
    for p, need, cost in cands:
        if spent + cost > total_wall:
            # shrink to what fits
            room = total_wall - spent
            if room <= 0:
                continue
            k = int(room / (MS_PER_SAMPLE.get(p["L"], 4000.0) / 1000.0) // 256) * 256
            if k < 256:
                continue
            need, cost = k, k * MS_PER_SAMPLE.get(p["L"], 4000.0) / 1000.0
        chosen.append((p, need, cost))
        spent += cost
    return chosen, spent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=3600.0,
                    help="wall-clock budget in seconds (per worker)")
    ap.add_argument("--run", type=int, default=0, help="launch N parallel workers")
    ap.add_argument("--tgt-small", type=float, default=0.002)
    ap.add_argument("--tgt-large", type=float, default=0.006)
    args = ap.parse_args()

    pts = read_points()
    chosen, spent = plan(pts, args.budget, max(args.run, 1),
                         (args.tgt_small, args.tgt_large))

    print(f"{'L':>4} {'beta':>8} {'z':>8} {'n':>6} {'D':>9} {'sem':>8} "
          f"{'+n':>6} {'cost_s':>8}")
    for p, need, cost in chosen:
        print(f"{p['L']:4d} {p['beta']:8.4f} {p['z']:8.3f} {p['n']:6d} "
              f"{p['mean']:9.5f} {p['sem']:8.5f} {need:6d} {cost:8.0f}")
    print(f"\ntotal CPU-seconds ~= {spent:.0f}  "
          f"({spent/3600:.2f} h);  wall with {max(args.run,1)} workers "
          f"~= {spent/max(args.run,1)/3600:.2f} h")

    if not args.run:
        return

    jobs = []
    for p, need, _ in chosen:
        jobs.append([sys.executable, os.path.join(HERE, "extend_samples.jl")])
    # launch via julia, chunked over workers
    def work(chunk):
        for p, need, _ in chunk:
            cmd = ["julia", "--project=@v1.11",
                   os.path.join(HERE, "extend_samples.jl"),
                   f"--file={p['file']}", f"--nextra={need}"]
            print("RUN", os.path.basename(p["file"]), need, flush=True)
            subprocess.run(cmd, cwd=HERE, check=False)

    n = args.run
    chunks = [chosen[i::n] for i in range(n)]
    with ThreadPoolExecutor(max_workers=n) as ex:
        list(ex.map(work, chunks))


if __name__ == "__main__":
    main()
