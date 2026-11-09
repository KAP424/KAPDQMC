"""Drive sharded extension runs across the worst error bars.

Unlike extend_errors.py (which appends to one CSV per job, so each point is
serial), this launches N shards per point in parallel via extend_shard.jl and
folds them in with merge_shards.py.  The expensive small-z points are the ones
that need it: L=30 at z=0.089 wants 34k samples = 23 CPU-h.

Usage:
    python run_shards.py --tol 0.05 --budget 34 --workers 14 --dry
    python run_shards.py --tol 0.05 --budget 34 --workers 14
"""

import argparse
import glob
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))

# measured single-process ms/sample with ~8 concurrent Julia processes
MS = {6: 0.4, 12: 9.0, 18: 78.0, 24: 600.0, 30: 2450.0, 36: 12000.0}
COLS = ("D", "SA", "SB", "SC")


def sem_rel(vals):
    n = len(vals)
    if n < 2:
        return float("inf")
    m = sum(vals) / n
    if m == 0:
        return float("inf")
    return (sum((v - m) ** 2 for v in vals) / (n - 1) / n) ** 0.5 / abs(m)


def read_csv(f):
    with open(f) as fh:
        lines = fh.read().splitlines()
    if len(lines) < 2:
        return None
    hdr = lines[0].split(",")
    ci = {c: hdr.index(c) for c in COLS}
    vals = {c: [] for c in COLS}
    for ln in lines[1:]:
        fld = ln.split(",")
        for c in COLS:
            vals[c].append(float(fld[ci[c]]))
    f0 = lines[1].split(",")
    return dict(file=f, L=int(float(f0[0])), beta=round(float(f0[1]), 4),
                z=float(f0[2]), n=len(vals["D"]), vals=vals)


def has_shards(parent):
    """True if this point already has shard files, i.e. a run is in flight.

    The planner must skip such points entirely.  Two concurrent runs would
    write shards with the same names from different seeds, and merge_shards.py
    would then fold in whichever set it happened to find.
    """
    return bool(glob.glob(parent + ".shard*of*.csv"))


def scan():
    """Pool every file sharing an (L, beta) before judging its error bar.

    A point may already be split across shard files
    (L30_beta4.6108.csv.shardKof14.csv).  Judging each file on its own sees a
    third of the statistics and reports an error bar three times too large, so
    the driver would keep re-buying samples the point already has.
    """
    pool = {}
    for d in ("data_matched_z", "data_gapfill"):
        for f in sorted(glob.glob(os.path.join(HERE, d, "*.csv"))):
            r = read_csv(f)
            if r is None:
                continue
            key = (r["L"], r["beta"])
            if key in pool:
                for c in COLS:
                    pool[key]["vals"][c].extend(r["vals"][c])
                pool[key]["n"] += r["n"]
            else:
                pool[key] = r
    pts = []
    for r in pool.values():
        pts.append(dict(file=r["file"], L=r["L"], z=r["z"], n=r["n"],
                        rel=max(sem_rel(r["vals"][c]) for c in COLS)))
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--budget", type=float, default=34.0, help="CPU-hours")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    pts = scan()
    jobs = []
    for p in pts:
        if p["rel"] <= args.tol:
            continue
        need = int(p["n"] * (p["rel"] / args.tol) ** 2)
        extra = (need // 256 + 1) * 256 - p["n"]
        # Skip points whose shards are still being written -- a second run
        # would reuse the same shard file names from a different seed.
        if has_shards(p["file"]):
            continue
        if extra < 256:
            continue
        ms = MS.get(p["L"], 12000.0)
        jobs.append((p, extra, extra * ms / 1000.0))
    jobs.sort(key=lambda j: -j[2])

    chosen, spent = [], 0.0
    for j in jobs:
        if spent + j[2] / 3600.0 > args.budget:
            continue
        chosen.append(j)
        spent += j[2] / 3600.0

    print(f"tol={args.tol:.3f}  budget={args.budget:.1f} CPU-h  "
          f"-> {len(chosen)} points, {spent:.1f} CPU-h, "
          f"wall >= {spent/max(args.workers,1):.1f} h")
    print(f"{'L':>3} {'z':>9} {'n':>6} {'rel%':>6} {'+':>6} {'shards':>6} "
          f"{'/shard':>7} {'CPU-h':>6}  file")
    plan = []
    for p, extra, cost in chosen:
        # Long jobs get more shards so every point finishes at roughly the same
        # time; a point costing < 0.3 CPU-h stays whole.
        nsh = 1
        if cost / 3600.0 > 0.3:
            nsh = min(args.workers, max(2, round(cost / 3600.0 * 3)))
            nsh = min(nsh, extra // 128)
        per = (extra // nsh // 64 + 1) * 64
        plan.append((p, nsh, per, cost))
        print(f"{p['L']:3d} {p['z']:9.3f} {p['n']:6d} {p['rel']*100:6.1f} "
              f"{extra:6d} {nsh:6d} {per:7d} {cost/3600:6.2f}  "
              f"{os.path.basename(p['file'])}")
    if args.dry:
        return

    logdir = os.path.join(HERE, "shard_logs")
    os.makedirs(logdir, exist_ok=True)

    def run(task):
        p, k, nsh, per = task
        log = os.path.join(logdir, f"{os.path.basename(p['file'])}.s{k}.log")
        cmd = ["julia", "--project=@v1.11",
               os.path.join(HERE, "extend_shard.jl"),
               f"--file={p['file']}", f"--shard={k}", f"--nshards={nsh}",
               f"--nextra={per}"]
        with open(log, "w") as fh:
            rc = subprocess.run(cmd, cwd=HERE, stdout=fh,
                                stderr=subprocess.STDOUT).returncode
        return p["file"], k, rc

    tasks = [(p, k, nsh, per) for p, nsh, per, _ in plan for k in range(nsh)]
    tasks.sort(key=lambda t: -t[3])          # longest shards first
    print(f"\nlaunching {len(tasks)} shards on {args.workers} workers")

    # Merge a point as soon as all of ITS shards are done, not at the very end.
    # The end-only version left finished points sitting in shard files for hours
    # while one slow point (L=30) kept running, and a crash lost them entirely.
    remaining = {p["file"]: nsh for p, nsh, _, _ in plan}
    finished = {p["file"]: 0 for p, nsh, _, _ in plan}

    def try_merge(f):
        if finished[f] == remaining[f]:
            r = subprocess.run(
                ["python", os.path.join(HERE, "merge_shards.py"),
                 "--force", os.path.basename(f)],
                cwd=HERE, capture_output=True, text=True)
            print(r.stdout.strip() or r.stderr.strip(), flush=True)

    t0 = time.time()
    done = 0
    bad = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run, t): t for t in tasks}
        for fu in as_completed(futs):
            f, k, rc = fu.result()
            done += 1
            if rc != 0:
                bad.append((f, k, rc))
                print(f"[{done}/{len(tasks)}] FAIL {os.path.basename(f)} "
                      f"shard {k} rc={rc}", flush=True)
            else:
                finished[f] += 1
                try_merge(f)
                if done % 5 == 0 or done == len(tasks):
                    print(f"[{done}/{len(tasks)}] {time.time()-t0:.0f}s", flush=True)
    print(f"\nall shards done in {(time.time()-t0)/3600:.2f} h, {len(bad)} failed")
    if bad:
        print("failed:", bad)
        print("shard files kept; rerun merge_shards.py after fixing")
    else:
        subprocess.run(["python", os.path.join(HERE, "merge_shards.py")],
                       cwd=HERE)


if __name__ == "__main__":
    main()
