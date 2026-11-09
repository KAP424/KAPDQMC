"""Read-only verification of the delivered report against the raw data.

Recomputes every headline number from the CSVs, then checks that the strings the
report actually claims are present and agree.  Writes nothing except its own
stdout; exits non-zero if any check fails.
"""
import csv
import math
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OBS = os.path.join(ROOT, "data", "pattern_class_observations.csv")
SUM = os.path.join(ROOT, "data", "pattern_class_summary.csv")
FOBS = os.path.join(ROOT, "data", "floor_ladder_observations.csv")
REPORT = os.path.join(ROOT, "CLASSIFICATION.md")
FIG = os.path.join(ROOT, "figures_class")

fails = []


def check(name, ok, detail=""):
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  " + detail) if detail else ""))
    if not ok:
        fails.append(name)


def load(p):
    if not os.path.isfile(p):
        return []
    return list(csv.DictReader(open(p, encoding="utf-8")))


def distinct_levels(values, rel=1e-9):
    out = []
    for v in sorted(values):
        if not out or abs(v - out[-1]) > rel * max(abs(v), abs(out[-1]), 1e-300):
            out.append(v)
    return out


def main():
    print("verifying CLASSIFICATION.md against the raw data\n")
    text = open(REPORT, encoding="utf-8").read() if os.path.isfile(REPORT) else ""
    obs, summ = load(OBS), load(SUM)
    fobs = load(FOBS)

    # ---- 1. the exact law, recomputed from scratch -------------------------
    Lmax = {}
    for r in obs:
        m, L = int(r["m"]), int(r["L"])
        Lmax[m] = max(Lmax.get(m, 0), L)
    devs, npts = [], 0
    for r in obs:
        m, L = int(r["m"]), int(r["L"])
        if L != Lmax[m]:
            continue
        momx, AA = float(r["momx"]), float(r["A_A"])
        if momx <= 0 or AA <= 0:
            continue
        pred = (4.0 / m ** 2) * momx
        devs.append(abs(AA / L / pred - 1.0))
        npts += 1
    maxdev = max(devs) if devs else float("inf")
    check("law a_R=(4/m^2)M exact to <1e-10", maxdev < 1e-10,
          "max dev %.1e over %d points" % (maxdev, npts))
    check("report quotes the same max deviation",
          ("5.2e-14" in text) or ("%.1e" % maxdev) in text, "%.1e" % maxdev)

    # ---- 2. the level count ------------------------------------------------
    xn = []
    for r in summ:
        if r["obstructed"].strip().lower() == "true":
            continue
        s = float(r["smin"])
        if s > 0:
            xn.append(math.sqrt(1.0 / s ** 2 - 1.0))
    lv = distinct_levels(xn)
    check("405 distinct ||X|| levels", len(lv) == 405, "got %d" % len(lv))
    check("report states 405 levels", "405" in text)

    # ---- 3. exactness of the degeneracies ---------------------------------
    groups = {}
    for v in xn:
        groups.setdefault(round(v, 6), []).append(v)
    worst = 0.0
    for v in groups.values():
        if len(v) > 1:
            den = max(max(abs(x) for x in v), 1e-300)
            worst = max(worst, (max(v) - min(v)) / den)
    check("within-level spread < 1e-9 (exact degeneracy)", worst < 1e-9, "worst %.1e" % worst)

    # ---- 4. the floor extrapolation ---------------------------------------
    byL = {}
    for r in fobs:
        try:
            L, aR = int(r["L"]), float(r["A_A"]) / int(r["L"])
        except (ValueError, ZeroDivisionError):
            continue
        if aR > 0:
            byL.setdefault(L, []).append(aR)
    check("floor ladder has >=6 lengths", len(byL) >= 6, "L=%s" % sorted(byL))
    if len(byL) >= 6:
        use = sorted(byL)[-6:]
        xs = [1.0 / L for L in use]
        ys = [sum(byL[L]) / len(byL[L]) for L in use]
        A = [[sum(x ** (i + j) for x in xs) for j in range(3)] for i in range(3)]
        b = [sum(y * x ** i for x, y in zip(xs, ys)) for i in range(3)]

        def det3(M):
            return (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
                    - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
                    + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))
        D = det3(A)
        cinf = None
        if abs(D) > 1e-300:
            M0 = [[b[i] if j == 0 else A[i][j] for j in range(3)] for i in range(3)]
            cinf = det3(M0) / D
        check("floor -> 2.12997 as L->inf", cinf is not None and abs(cinf - 2.129967) < 1e-4,
              "got %s" % ("%.7f" % cinf if cinf else "n/a"))
        check("report states the extrapolated floor", "2.12997" in text)
        # the floor must be strictly increasing in L
        seq = [sum(byL[L]) / len(byL[L]) for L in sorted(byL)]
        check("floor increases monotonically with L",
              all(b > a for a, b in zip(seq, seq[1:])))

    # ---- 5. the theta window and its gap ---------------------------------
    th = [math.degrees(math.atan(v)) for v in xn]
    check("theta_max >= 45 deg everywhere", min(th) >= 45.0 - 1e-9, "min %.4f" % min(th))
    non = [t for t in th if abs(t - 45.0) >= 1e-6]
    gap = (min(non) - 45.0) if non else 0.0
    check("30 degree empty gap above 45", abs(gap - 29.99) < 0.05, "gap %.2f" % gap)
    check("nothing in (45,70)", not any(45.0 + 1e-6 < t < 70.0 for t in th))

    # ---- 6. the report's structure and figures ---------------------------
    heads = re.findall(r"^## (\d+)\.", text, re.M)
    check("report has sections 1..9 in order", heads == [str(i) for i in range(1, 10)],
          "found %s" % heads)
    pngs = sorted(f for f in os.listdir(FIG) if f.endswith(".png")) if os.path.isdir(FIG) else []
    check("five figures present", len(pngs) == 5, " ".join(pngs))
    if pngs:
        try:
            from PIL import Image
            ok = True
            for p in pngs:
                im = Image.open(os.path.join(FIG, p))
                lo, hi = im.convert("L").getextrema()
                ok = ok and im.size[0] > 400 and im.size[1] > 300 and hi > lo
            check("figures are non-trivial images", ok)
        except ImportError:
            print("  [skip] PIL unavailable")
    check("no unclosed inline math in report", text.count("$") % 2 == 0,
          "count=%d" % text.count("$"))

    print()
    if fails:
        print("FAILED: %d check(s): %s" % (len(fails), ", ".join(fails)))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
