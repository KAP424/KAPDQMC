"""Fold extend_shard.jl outputs back into the parent CSV.

Each shard file is `NAME.shardKofN.csv` next to `NAME`.  This appends every
shard's rows to NAME (renumbering the `sample` column so it stays unique), then
deletes the shard files.  Refuses to run if a shard is short of the requested
row count unless --force is given, so a killed shard is not silently merged.

Usage:
    python merge_shards.py                 # merge every complete set
    python merge_shards.py --dry           # show what would merge
    python merge_shards.py --force L30_beta4.6108.csv   # merge a partial set
"""

import argparse
import csv
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAT = re.compile(r"^(?P<base>.+?)\.shard(?P<k>\d+)of(?P<n>\d+)\.csv$")


def collect():
    """Group shard files by their parent CSV path."""
    groups = {}
    for f in glob.glob(os.path.join(HERE, "data_*", "*.shard*of*.csv")):
        mt = PAT.match(os.path.basename(f))
        if not mt:
            continue
        parent = os.path.join(os.path.dirname(f), mt.group("base"))
        groups.setdefault(parent, []).append(
            (int(mt.group("k")), int(mt.group("n")), f))
    return groups


def merge(parent, shards, force=False, dry=False):
    shards.sort()
    nshard = shards[0][1]
    if len(shards) != nshard:
        msg = f"{len(shards)}/{nshard} shards present"
        if not force:
            return f"SKIP {os.path.basename(parent)}: {msg} (use --force)"
    # every shard must have the same row count
    counts = []
    for _, _, f in shards:
        with open(f) as fh:
            counts.append(sum(1 for _ in fh) - 1)
    if len(set(counts)) != 1 and not force:
        return (f"SKIP {os.path.basename(parent)}: ragged shards {counts} "
                f"(use --force)")

    with open(parent) as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    n0 = len(body)
    added = 0
    # The sample column must continue from the largest existing index, not from
    # the row count: a gapfill parent can be numbered 257..512, and n0 + added
    # would restart at 257 and collide.
    next_index = max((int(r[3]) for r in body if len(r) > 3), default=0) + 1
    # Column width.  extend_shard.jl always writes 11 columns, but two parents
    # in data_matched_z carry a 10-column header (no elapsed field).  Appending
    # 11-field rows under a 10-field header makes np.genfromtxt(names=True) --
    # which every figure and headline script uses -- raise on the first such
    # row, taking the whole analysis down.
    ncol = len(header)
    narrowed = 0
    # elapsed: every shard restarts its clock at 0, so copying the field
    # unchanged made the merged column non-monotone and its last value only the
    # LAST shard's time.  Anything that reads elapsed[-1]/n as a per-sample cost
    # (extend_errors.py's MS table was calibrated that way) would come out too
    # small by about the number of shards.  Offset each shard by the running
    # total so the column is cumulative CPU seconds, as extend_samples.jl writes
    # it.  Files merged before this fix keep shard-relative values; do not
    # recalibrate costs from them.
    ie = header.index("elapsed") if "elapsed" in header else None
    t_base = 0.0
    if ie is not None and body and len(body[-1]) > ie:
        try:
            t_base = float(body[-1][ie])
        except ValueError:
            t_base = 0.0
    for _, _, f in shards:
        with open(f) as fh:
            r = list(csv.reader(fh))
        t_last = 0.0
        for row in r[1:]:
            if len(row) > ncol:
                row = row[:ncol]
                narrowed += 1
            elif len(row) < ncol:
                return (f"SKIP {os.path.basename(parent)}: shard row has "
                        f"{len(row)} fields, parent header has {ncol}")
            row[3] = str(next_index + added)
            if ie is not None:
                t_last = float(row[ie])
                row[ie] = f"{t_base + t_last:.2f}"
            body.append(row)
            added += 1
        t_base += t_last

    if dry:
        # --dry must not touch the filesystem: an earlier version parsed the
        # flag but never used it, so a "preview" merged and deleted the shards.
        return (f"WOULD MERGE {os.path.basename(parent)}: {n0} -> "
                f"{n0 + added} samples"
                + (f" ({narrowed} rows truncated to {ncol} cols)" if narrowed else ""))

    with open(parent, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(body)
    for _, _, f in shards:
        os.remove(f)
    return (f"OK {os.path.basename(parent)}: {n0} -> {n0 + added} samples"
            + (f" ({narrowed} rows truncated to {ncol} cols)" if narrowed else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--force", nargs="*", default=None,
                    metavar="CSV",
                    help="merge even if shards are missing/ragged.  With no "
                         "names, applies to every group; with names, only those")
    args = ap.parse_args()

    groups = collect()
    if not groups:
        print("no shard files found")
        return
    for parent in sorted(groups):
        base = os.path.basename(parent)
        # --force with names restricts the run to those files; --force alone
        # forces every group.  Plain (no --force) checks completeness.
        if args.force:
            if base not in args.force:
                continue
        msg = merge(parent, groups[parent], force=args.force is not None,
                    dry=args.dry)
        print(msg)


if __name__ == "__main__":
    main()
