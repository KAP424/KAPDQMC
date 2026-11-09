"""Referee (I)-a3: re-do the blind prediction with several training/test splits; chi2 per split."""
import sys; sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030/docs/audit")
import numpy as np, ref2_common as c
pts, base = c.load()
groups = []
for p in sorted(pts.values(), key=lambda p: p["z"]):
    for g in groups:
        if abs(g[0]["z"] - p["z"]) / p["z"] < 3e-3: g.append(p); break
    else: groups.append([p])
groups = [sorted(g, key=lambda q: q["L"]) for g in groups if len(g) >= 3 and g[0]["z"] < 20]
def split(train_max, test_min, test_max=99, label=""):
    pulls = []; det = []
    for g in groups:
        tr = [q for q in g if q["L"] <= train_max]; te = [q for q in g if test_min <= q["L"] <= test_max]
        if not tr or not te: continue
        m, e = c.wmean([q["mean"] for q in tr], [q["sem"] for q in tr])
        for q in te:
            pull = (q["mean"] - m) / np.hypot(q["sem"], e); pulls.append(pull); det.append((g[0]["z"], q["L"], pull))
    pulls = np.array(pulls)
    print("  %-34s npred=%2d  chi2/dof=%.2f  rms=%.2f  max|pull|=%.2f  mean pull=%+.2f (sign test: %d neg / %d)" % (
        label, len(pulls), (pulls ** 2).mean(), np.sqrt((pulls ** 2).mean()), np.abs(pulls).max(), pulls.mean(), (pulls < 0).sum(), len(pulls)))
    return det
print("Blind prediction  D_pred(L,z) = weighted mean of training L at the same z")
d1 = split(24, 30, label="train L<=24 -> test L=30,36 (summary)")
split(18, 24, label="train L<=18 -> test L=24,30,36")
split(12, 18, label="train L<=12 -> test L=18..36")
split(6, 12, label="train L=6    -> test L=12..36")
split(18, 24, 24, label="train L<=18 -> test L=24")
split(24, 30, 30, label="train L<=24 -> test L=30")
split(30, 36, 36, label="train L<=30 -> test L=36")
print("  details (train<=24, test 30/36):", [(round(a, 2), b, round(cc, 2)) for a, b, cc in d1])
print("  new gapfill points z=5.93,10.55 as blind tests of the L=18/24 reference: pulls", [(round(a, 2), b, round(cc, 2)) for a, b, cc in split(24, 30, label="(incl. z=5.93/10.55 rows)") if a > 5])
