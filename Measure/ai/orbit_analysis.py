"""Is ||X|| a function of the translation x complement orbit?

Motivation.  Two operations are proved to preserve ||X||:

  * supercell translation T(dx,dy):  u_k[T(F),:] = diag(e^{ik.r}) u_k[F,:], a
    diagonal similarity, so sigma_min(u_k[F,:]) is unchanged;
  * complement (E <-> F, i.e. bitwise NOT of the half-filled pattern).

So the orbit under G = translations (m^2 elements) x complement (order 2) is a
LOWER bound on the invariant that fixes ||X||.  If the number of G-orbits equals
the number of distinct ||X|| levels inside each m, and no orbit is split by ||X||,
then ||X|| is exactly an orbit invariant and the classification is complete:
configurations are classified by their translation/complement orbit.

Reads only the measured CSVs; writes nothing.
"""
import csv
import math
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
SUM = os.path.join(ROOT, "data", "pattern_class_summary.csv")


def load_summary():
    out = []
    if not os.path.isfile(SUM):
        return out
    for r in csv.DictReader(open(SUM, encoding="utf-8")):
        if r["obstructed"].strip().lower() == "true":
            continue
        try:
            smin = float(r["smin"])
        except ValueError:
            continue
        if smin <= 0:
            continue
        out.append(dict(m=int(r["m"]), label=r["label"], cls=int(r["cls"]),
                        bits=r["label"].split("_", 1)[1],
                        xnorm=math.sqrt(1.0 / smin ** 2 - 1.0)))
    return out


def translate(bits, m, dx, dy):
    out = list(bits)
    for y in range(m):
        for x in range(m):
            sx, sy = (x - dx) % m, (y - dy) % m
            for s in (0, 1):
                out[2 * (x + m * y) + s] = bits[2 * (sx + m * sy) + s]
    return "".join(out)


def complement(bits):
    return "".join("0" if c == "1" else "1" for c in bits)


def group_images(bits, m):
    seen = set()
    for dx in range(m):
        for dy in range(m):
            t = translate(bits, m, dx, dy)
            seen.add(t)
            seen.add(complement(t))
    return seen


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def distinct_levels(values, rel=1e-9):
    out = []
    for v in sorted(values):
        if not out or abs(v - out[-1]) > rel * max(abs(v), abs(out[-1]), 1e-300):
            out.append(v)
    return out


def analyse(rows, m, verbose=True):
    sub = [r for r in rows if r["m"] == m]
    if not sub:
        return None
    index = {r["bits"]: i for i, r in enumerate(sub)}
    dsu = DSU(len(sub))
    for i, r in enumerate(sub):
        for img in group_images(r["bits"], m):
            j = index.get(img)
            if j is not None:
                dsu.union(i, j)
    orbits = defaultdict(list)
    for i in range(len(sub)):
        orbits[dsu.find(i)].append(i)

    levels = distinct_levels([r["xnorm"] for r in sub])
    # does every orbit sit on a single level?
    split = 0
    worst = 0.0
    for members in orbits.values():
        xs = [sub[i]["xnorm"] for i in members]
        den = max(max(abs(v) for v in xs), 1e-300)
        spread = (max(xs) - min(xs)) / den
        worst = max(worst, spread)
        if spread > 1e-6:
            split += 1
    # does every level correspond to exactly one orbit?
    levels_of = defaultdict(set)
    for members in orbits.values():
        key = round(sub[members[0]]["xnorm"], 6)
        for i in members:
            levels_of[round(sub[i]["xnorm"], 6)].add(dsu.find(members[0]))
    multi_orbit_levels = sum(1 for k, v in levels_of.items() if len(v) > 1)

    if verbose:
        print("m=%d" % m)
        print("   patterns            : %d" % len(sub))
        print("   G-orbits (trans x ~) : %d" % len(orbits))
        print("   distinct ||X|| levels: %d" % len(levels))
        print("   orbits split by ||X||: %d   (worst within-orbit spread %.2e)"
              % (split, worst))
        print("   levels hit by >1 orbit: %d" % multi_orbit_levels)
        print("   orbit sizes: %s" % sorted((len(v) for v in orbits.values()), reverse=True)[:8])
        print()
    return dict(m=m, n=len(sub), orbits=len(orbits), levels=len(levels),
                split=split, worst=worst, multi=multi_orbit_levels)


def main():
    rows = load_summary()
    if not rows:
        print("no data; run .auto/measure.sh first")
        return
    print("Testing whether ||X|| is a function of the translation x complement orbit")
    print("(translation and complement are both proved to preserve ||X||)\n")
    res = []
    for m in sorted({r["m"] for r in rows}):
        r = analyse(rows, m)
        if r:
            res.append(r)
    print("verdict:")
    ok = all(r["split"] == 0 and r["orbits"] == r["levels"] for r in res)
    for r in res:
        verdict = ("EXACT" if r["split"] == 0 and r["orbits"] == r["levels"]
                   else "not exact")
        print("  m=%d: %d orbits vs %d levels -> %s" % (r["m"], r["orbits"], r["levels"], verdict))
    print()
    if ok:
        print("||X|| is exactly an orbit invariant on every m measured: the")
        print("classification of configurations IS the translation x complement orbit.")
    else:
        print("||X|| is NOT exactly an orbit invariant: the orbits are a strict")
        print("coarsening (or a split), so a finer discrete invariant is at work.")
    print()
    print("METRIC pc_orbit_exact=%d" % (1 if ok else 0))


if __name__ == "__main__":
    main()
