"""Shared loader for the hostile-referee audit (ref2_*). Pools samples per (L, beta)."""
import glob, json, os
from collections import defaultdict
import numpy as np

R = "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030"
AUD = R + "/docs/audit"


def load():
    base = json.load(open(R + "/baseline_D.json"))
    pool = defaultdict(list)
    zv = {}
    for d in ("data_matched_z", "data_matched_largez", "data_gapfill"):
        for f in sorted(glob.glob(f"{R}/{d}/*.csv")):
            a = np.genfromtxt(f, delimiter=",", names=True)
            if a.size == 0:
                continue
            key = (int(a["L"][0]), round(float(a["beta"][0]), 4))
            pool[key].append(a)
            zv[key] = float(a["z"][0])
    pts = {}
    for k, arrs in sorted(pool.items()):
        cols = {c: np.concatenate([np.atleast_1d(x[c]) for x in arrs]) for c in ("D", "SA", "SB", "SC", "DV", "j")}
        D = cols["D"]
        pts[k] = dict(L=k[0], beta=k[1], z=zv[k], n=D.size, mean=D.mean(),
                      sem=D.std(ddof=1) / np.sqrt(D.size), Dgs=base[str(k[0])]["D"],
                      nfiles=len(arrs), cols=cols)
    return pts, base


def wmean(m, s):
    m = np.asarray(m, float); w = 1 / np.asarray(s, float) ** 2
    return float((w * m).sum() / w.sum()), float(1 / np.sqrt(w.sum()))
