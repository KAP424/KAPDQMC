"""Launch parallel sample-extension jobs for the 2D production grid.

Each (L, beta) point is a separate Julia process (BLAS pinned to 1 thread), so
the parallelism is across points.  Measured single-sample cost on this machine:

    L=6  0.1 ms    L=12 3.2 ms    L=18 66 ms    L=24 246 ms
    L=30 810 ms    L=36 3912 ms

Targets are chosen to bring the absolute SEM on D to roughly 0.002 for L<=24
and 0.004 for L>=30, which is where the figures stop being dominated by error
bars.

Usage:
    python extend_run.py --plan                 # print what would run
    python extend_run.py --jobs 12              # launch
    python extend_run.py --jobs 12 --only large # only L>=30
"""

import argparse
import csv
import glob
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))

MS = {6: 0.1, 12: 3.2, 18: 66.0, 24: 246.0, 30: 810.0, 36: 3912.0}

# target total samples per point.  Chosen from the measured cost curve so the
# whole sweep fits in roughly one wall-clock hour on 13 workers:
#   L=30  -> sem ~ 0.005 on D ~ 0.5   (was 0.020 with 512 samples)
#   L=36  -> sem ~ 0.006 on D ~ 0.65  (was 0.025 with 512 samples)
TARGET = {6: 20_000, 12: 20_000, 18: 8_000, 24: 6_000, 30: 3_000, 36: 1_400}
# gapfill files hold 256 samples each at large z; same targets by L
GAPFILL_EXTRA = {30: 2_800, 36: 1_200}


def scan():
    pts = []
    for f in sorted(glob.glob(os.path.join(HERE, "data_matched_z", "*.csv"))):
        with open(f) as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            continue
        L = int(float(rows[0]["L"]))
        d = [float(r["D"]) for r in rows]
        pts.append(dict(file=f, L=L, beta=float(rows[0]["beta"]),
                        z=float(rows[0]["z"]), n=len(d),
                        target=TARGET.get(L, 2000), kind="matched"))
    for f in sorted(glob.glob(os.path.join(HERE, "data_gapfill", "*.csv"))):
        with open(f) as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            continue
        L = int(float(rows[0]["L"]))
        pts.append(dict(file=f, L=L, beta=float(rows[0]["beta"]),
                        z=float(rows[0]["z"]), n=len(rows),
                        target=GAPFILL_EXTRA.get(L, 1500), kind="gapfill"))
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--only", choices=["all", "large", "small"], default="all")
    args = ap.parse_args()

    pts = scan()
    if args.only == "large":
        pts = [p for p in pts if p["L"] >= 30]
    elif args.only == "small":
        pts = [p for p in pts if p["L"] <= 24]

    jobs = []
    for p in pts:
        extra = max(0, p["target"] - p["n"])
        extra = (extra // 256) * 256
        if extra < 256:
            continue
        cost = extra * MS.get(p["L"], 4000.0) / 1000.0
        jobs.append((p, extra, cost))

    jobs.sort(key=lambda j: -j[2])          # longest first for better packing
    total = sum(c for _, _, c in jobs)
    print(f"{len(jobs)} jobs, {total/3600:.2f} CPU-hours, "
          f"wall >= {total/max(args.jobs,1)/3600:.2f} h with {args.jobs} workers")
    for p, extra, cost in jobs:
        print(f"  L={p['L']:2d} beta={p['beta']:7.4f} z={p['z']:8.3f} "
              f"n={p['n']:5d} +{extra:5d}  ~{cost/60:6.1f} min  "
              f"{os.path.basename(p['file'])}")
    if args.plan:
        return

    logdir = os.path.join(HERE, "extend_logs")
    os.makedirs(logdir, exist_ok=True)

    def run(job):
        p, extra, _ = job
        log = os.path.join(logdir, os.path.basename(p["file"]) + ".log")
        cmd = ["julia", "--project=@v1.11",
               os.path.join(HERE, "extend_samples.jl"),
               f"--file={p['file']}", f"--nextra={extra}"]
        t0 = time.time()
        with open(log, "w") as fh:
            rc = subprocess.run(cmd, cwd=HERE, stdout=fh,
                                stderr=subprocess.STDOUT).returncode
        return p["file"], extra, rc, time.time() - t0

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run, j): j for j in jobs}
        for fu in as_completed(futs):
            f, extra, rc, dt = fu.result()
            done += 1
            print(f"[{done}/{len(jobs)}] {os.path.basename(f)} +{extra} "
                  f"rc={rc} {dt/60:.1f} min", flush=True)
    print(f"all done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
