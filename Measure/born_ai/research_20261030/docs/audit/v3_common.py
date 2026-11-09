"""Independent loader for the v3 audit (does not import any project script)."""
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

R = "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030"
AUD = R + "/docs/audit"
DIRS = ("data_matched_z", "data_matched_largez", "data_gapfill")


def read_csv(path):
    with open(path, newline="") as fh:
        rd = csv.reader(fh)
        hdr = next(rd)
        rows = [r for r in rd if r]
    arr = np.array([[float(x) for x in r] for r in rows])
    return hdr, arr


def load_all():
    """returns dict key=(L,beta4) -> dict(D=..., DV=..., SA, SB, SC, j, z, files=[...])"""
    pool = defaultdict(lambda: defaultdict(list))
    zs = defaultdict(set)
    files = defaultdict(list)
    for d in DIRS:
        for f in sorted(glob.glob(f"{R}/{d}/*.csv")):
            hdr, a = read_csv(f)
            if a.shape[0] == 0:
                continue
            col = {h: i for i, h in enumerate(hdr)}
            L = int(a[0, col["L"]])
            b = round(float(a[0, col["beta"]]), 4)
            key = (L, b)
            for nm in ("j", "SA", "SB", "SC", "D", "DV"):
                pool[key][nm].append(a[:, col[nm]])
            zs[key].add(round(float(a[0, col["z"]]), 4))
            files[key].append((d, os.path.basename(f), a.shape[0]))
    out = {}
    for key, dd in pool.items():
        e = {nm: np.concatenate(v) for nm, v in dd.items()}
        e["z"] = sorted(zs[key])
        e["files"] = files[key]
        e["n"] = e["D"].size
        e["mean"] = float(e["D"].mean())
        e["sem"] = float(e["D"].std(ddof=1) / np.sqrt(e["D"].size))
        out[key] = e
    return out


def baseline():
    return json.load(open(R + "/baseline_D.json"))
