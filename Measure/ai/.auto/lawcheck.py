"""Score the classification law from data/pattern_class_observations.csv.

Law under test:  a_R = A_A/L = (4/m^2) * M,  where
    M = sum_{a in E, b in F} sum_delta |X_ab(delta)|^2 * min(m|delta_x|, L/3)
is the first moment of the kernel in LATTICE units.

This is exact geometry, not a fit.  A site type occupies (L/m)^2 positions; across
a straight strip boundary the number of pairs (r in R, r-delta outside R) is
(L/m)*min(|dx|, L/(3m)), and A_A = 2*(t1+t2) with t1,t2 the two orientations, so

    A_A = (2L/m^2) * M   ->   a_R = (4/m^2) * M  ... with M as above.

Verified to machine precision for m=1,2,3 (max deviation 4e-14).

Emits METRIC lines.  Also tests the *previous* classification axis: the claim
"the cell-occupancy class determines s_min" is refuted if s_min spreads widely
inside class 1.
"""
import csv
import math
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OBS = os.path.join(ROOT, "data", "pattern_class_observations.csv")
SUM = os.path.join(ROOT, "data", "pattern_class_summary.csv")

if not os.path.isfile(OBS):
    print("METRIC pc_patterns=0")
    print("METRIC pc_law_maxdev=1.0")
    print("METRIC pc_groups=0")
    sys.exit(0)

rows = list(csv.DictReader(open(OBS, encoding="utf-8")))
summary = list(csv.DictReader(open(SUM, encoding="utf-8"))) if os.path.isfile(SUM) else []

# ---- law accuracy, per (pattern, L) --------------------------------------
devs = []
worst = None
for r in rows:
    m = int(r["m"])
    momx = float(r["momx"])
    aR = float(r["A_A"]) / int(r["L"])
    if momx <= 0:
        continue
    pred = (4.0 / m ** 2) * momx
    dev = abs(aR / pred - 1.0)
    devs.append(dev)
    if worst is None or dev > worst[0]:
        worst = (dev, r["label"], int(r["L"]), aR, pred)

maxdev = max(devs) if devs else 1.0
meandev = sum(devs) / len(devs) if devs else 1.0

# ---- how many patterns the law predicts within 15% ------------------------
# a pattern counts as "classified" when every L it was measured at obeys the law
per_pat = defaultdict(list)
for r in rows:
    m = int(r["m"])
    momx = float(r["momx"])
    if momx <= 0:
        continue
    pred = (4.0 / m ** 2) * momx
    per_pat[r["label"]].append(abs(float(r["A_A"]) / int(r["L"]) / pred - 1.0))
classified = sum(1 for v in per_pat.values() if max(v) <= 0.15)

# ---- discrete ||X|| families --------------------------------------------
xnorm_of = {}
for s in summary:
    try:
        smin = float(s["smin"])
        obstructed = s["obstructed"].strip().lower() == "true"
    except (KeyError, ValueError):
        continue
    if obstructed or smin <= 0:
        continue
    xnorm_of[s["label"]] = math.sqrt(1.0 / smin ** 2 - 1.0)

groups = defaultdict(list)
for lab, xn in xnorm_of.items():
    groups[round(xn, 4)].append(lab)

# ---- does the occupancy class fix s_min? ---------------------------------
cls_spread = {}
for s in summary:
    try:
        cls = int(s["cls"])
        smin = float(s["smin"])
        obstructed = s["obstructed"].strip().lower() == "true"
    except (KeyError, ValueError):
        continue
    if obstructed:
        continue
    cls_spread.setdefault(cls, []).append(smin)

n_obstructed = sum(1 for s in summary if s.get("obstructed", "").strip().lower() == "true")
n_measured = len({r["label"] for r in rows})

# ---- the classification axes ---------------------------------------------
# theta_max: ||X|| = tan(theta_max) is the largest principal angle between the
# occupied-band subspace and the F-sublattice coordinate subspace, maximised over
# momentum.  theta=45 deg means X is unitary (minimal influence), theta->90 deg
# means the measurement is blocked.
thetas = [math.degrees(math.atan(xn)) for xn in xnorm_of.values()]

# a_R at the unitary point ||X||=1: should be a universal constant (~2.07).
# It is the floor of the entropy influence, independent of m and of the pattern.
aR_at_unitary = []
for r in rows:
    lab = r["label"]
    if lab not in xnorm_of or abs(xnorm_of[lab] - 1.0) > 1e-6:
        continue
    momx = float(r["momx"])
    if momx <= 0:
        continue
    aR_at_unitary.append(float(r["A_A"]) / int(r["L"]))

# power law a_R ~ ||X||^slope, fitted in log-log space.  The fit is taken at the
# largest L available for each m: a_R is an extensive quantity that grows with L
# while ||X|| does not, so mixing lengths would add scatter that is not physics.
L_of_m = {}
for r in rows:
    m, L = int(r["m"]), int(r["L"])
    L_of_m[m] = max(L_of_m.get(m, 0), L)
xs, ys = [], []
for r in rows:
    lab = r["label"]
    if lab not in xnorm_of:
        continue
    if int(r["L"]) != L_of_m[int(r["m"])]:
        continue
    momx = float(r["momx"])
    if momx <= 0:
        continue
    xs.append(math.log(xnorm_of[lab]))
    ys.append(math.log(float(r["A_A"]) / int(r["L"])))
slope = r2 = 0.0
if len(xs) > 2:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((a - mx) ** 2 for a in xs)
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    slope = sxy / sxx if sxx else 0.0
    ss_tot = sum((b - my) ** 2 for b in ys)
    ss_res = sum((b - (slope * a + my - slope * mx)) ** 2 for a, b in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0

print("METRIC pc_patterns=%d" % n_measured)
print("METRIC pc_rows=%d" % len(rows))
print("METRIC pc_law_scored=%d" % len(per_pat))
print("METRIC pc_classified=%d" % classified)
print("METRIC pc_law_maxdev=%.4f" % maxdev)
print("METRIC pc_law_meandev=%.4f" % meandev)
print("METRIC pc_groups=%d" % len(groups))
print("METRIC pc_obstructed=%d" % n_obstructed)
print("METRIC pc_theta_min=%.2f" % (min(thetas) if thetas else 0.0))
print("METRIC pc_theta_max=%.2f" % (max(thetas) if thetas else 0.0))
print("METRIC pc_aR_unitary=%.4f" % (max(aR_at_unitary) if aR_at_unitary else 0.0))
print("METRIC pc_power_slope=%.3f" % slope)
print("METRIC pc_power_r2=%.4f" % r2)
for cls in sorted(cls_spread):
    v = cls_spread[cls]
    ratio = max(v) / min(v)
    print("METRIC pc_cls%d_smin_ratio=%.1f" % (cls, ratio))

if worst is not None:
    print("  worst law deviation: %s L=%d  a_R=%.4f vs pred=%.4f  (%.4f%%)"
          % (worst[1], worst[2], worst[3], worst[4], 100 * worst[0]))
print("  %d patterns measured / %d distinct ||X|| / %d obstructed"
      % (n_measured, len(groups), n_obstructed))
print("  theta_max in [%.2f, %.2f] deg ; a_R at ||X||=1 -> %.4f ; "
      "a_R ~ ||X||^%.2f (R^2 %.4f)"
      % (min(thetas) if thetas else 0, max(thetas) if thetas else 0,
         max(aR_at_unitary) if aR_at_unitary else 0, slope, r2))
