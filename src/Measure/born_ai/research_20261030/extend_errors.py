"""Top up 2D sample counts where the error bars are actually large.

extend_run.py targets a fixed sample count per L.  That is the wrong objective:
the visible error bar in the figures is the *relative* standard error, and at
small z the signal (D ~ z^2) collapses while the noise does not, so the
small-z points need far more samples than the large-z ones at the same L.

This script ranks every point by its measured relative SEM and buys samples
only where the figure still shows the bar:

    n_needed = n_have * (sem_rel / tol_rel)^2

with a per-point cost from the measured ms/sample curve, so a fixed CPU budget
buys the largest reduction in the worst bars.

Usage:
    python extend_errors.py --tol 0.05 --budget 3.0 --plan
    python extend_errors.py --tol 0.05 --budget 3.0 --jobs 13
"""

import argparse
import csv
import glob
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))

# ms per sample, calibrated from the `elapsed` column of the existing CSVs
# (median over files; the old model was 3-4x too optimistic at large L).
MS = {6: 0.2, 12: 3.7, 18: 42.6, 24: 495.0, 30: 2303.0, 36: 11629.0}
COLS = ("D", "SA", "SB", "SC")


def sem_rel(vals):
    n = len(vals)
    if n < 2:
        return float("inf")
    m = sum(vals) / n
    if m == 0:
        return float("inf")
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return (var / n) ** 0.5 / abs(m)


def busy(path):
    """True if some other runner is currently appending to this CSV.

    extend_samples.jl appends in place, so two processes writing the same file
    would interleave rows and duplicate sample indices.  A stale log without a
    DONE line, newer than the CSV, means a writer is (or was) active.
    """
    log = os.path.join(HERE, "extend_logs", os.path.basename(path) + ".log")
    if not os.path.exists(log):
        return False
    try:
        with open(log) as fh:
            txt = fh.read()
    except OSError:
        return True
    if "DONE " in txt:
        return False
    return os.path.getmtime(log) >= os.path.getmtime(path)


def scan():
    """One entry per (L, beta), pooling every file that shares the key.

    A point can already be split across shard files
    (L30_beta4.6108.csv.shardKof14.csv).  Judging each file separately sees a
    fraction of the statistics and reports an error bar sqrt(Nshards) times too
    large, so this planner would keep buying samples the point already has --
    which is exactly how the L=30 small-z rows got re-queued repeatedly.

    The entry keeps the PARENT file name, because that is what extend_samples.jl
    appends to and what `busy()` looks for.
    """
    pool = {}
    for d in ("data_matched_z", "data_gapfill"):
        for f in sorted(glob.glob(os.path.join(HERE, d, "*.csv"))):
            base = re.sub(r"\.shard\d+of\d+\.csv$", "", f)
            with open(f) as fh:
                rows = list(csv.DictReader(fh))
            if not rows:
                continue
            L = int(float(rows[0]["L"]))
            key = (L, round(float(rows[0]["beta"]), 4))
            e = pool.get(key)
            if e is None:
                e = pool[key] = dict(file=base, L=L,
                                     beta=float(rows[0]["beta"]),
                                     z=float(rows[0]["z"]),
                                     cols={c: [] for c in COLS})
            for r in rows:
                for c in COLS:
                    e["cols"][c].append(float(r[c]))
    pts = []
    for e in pool.values():
        rel = {c: sem_rel(v) for c, v in e["cols"].items()}
        pts.append(dict(file=e["file"], L=e["L"], beta=e["beta"], z=e["z"],
                        n=len(e["cols"]["D"]), rel=max(rel.values()),
                        busy=busy(e["file"])))
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=0.05,
                    help="target relative SEM on the worst of D/SA/SB/SC")
    ap.add_argument("--budget", type=float, default=3.0,
                    help="CPU-hours to spend")
    ap.add_argument("--jobs", type=int, default=13)
    ap.add_argument("--cap", type=float, default=2.0,
                    help="max CPU-hours for any single job; a point whose bar "
                         "would need more than this is only partially improved")
    ap.add_argument("--plan", action="store_true")
    args = ap.parse_args()

    pts = scan()
    nbusy = sum(1 for p in pts if p["busy"])
    if nbusy:
        print(f"skipping {nbusy} files another runner is still appending to")
    jobs = []
    for p in pts:
        if p["busy"] or p["rel"] <= args.tol:
            continue
        need = p["n"] * (p["rel"] / args.tol) ** 2
        extra = int(need) - p["n"]
        extra = (extra // 256) * 256
        if extra < 256:
            continue
        ms = MS.get(p["L"], 12000.0)
        cap_n = int(args.cap * 3600 * 1000 / ms)
        cap_n = (cap_n // 256) * 256
        extra = min(extra, cap_n)
        if extra < 256:
            continue
        jobs.append((p, extra, extra * ms / 1000.0))

    # Worst error bar first.  Buying the cheapest job first is wrong: it spends
    # the whole budget on points that are already at 4-5% and leaves the 46% and
    # 25% bars untouched.  Fix the bars the figure actually shows.
    jobs.sort(key=lambda j: -j[0]["rel"])
    chosen, spent = [], 0.0
    for j in jobs:
        if spent + j[2] / 3600.0 > args.budget:
            continue
        chosen.append(j)
        spent += j[2] / 3600.0
    chosen.sort(key=lambda j: -j[2])

    print(f"tol={args.tol:.3f}  budget={args.budget:.2f} CPU-h  "
          f"-> {len(chosen)} jobs, {spent:.2f} CPU-h, "
          f"wall >= {spent/max(args.jobs,1):.2f} h with {args.jobs} workers")
    print(f"{'L':>3} {'beta':>7} {'z':>9} {'n':>6} {'rel%':>6} "
          f"{'+':>6} {'->rel%':>7} {'min':>6}  file")
    for p, extra, cost in chosen:
        newrel = p["rel"] * (p["n"] / (p["n"] + extra)) ** 0.5 * 100
        print(f"{p['L']:3d} {p['beta']:7.4f} {p['z']:9.3f} {p['n']:6d} "
              f"{p['rel']*100:6.1f} {extra:6d} {newrel:7.1f} "
              f"{cost/60:6.1f}  {os.path.basename(p['file'])}")
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
        futs = {ex.submit(run, j): j for j in chosen}
        for fu in as_completed(futs):
            f, extra, rc, dt = fu.result()
            done += 1
            print(f"[{done}/{len(chosen)}] {os.path.basename(f)} +{extra} "
                  f"rc={rc} {dt/60:.1f} min", flush=True)
    print(f"all done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
