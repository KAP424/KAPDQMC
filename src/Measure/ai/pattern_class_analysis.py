"""Classify half-filled configurations by their post-measurement entropy influence.

Inputs (written by pattern_classes.jl):
    data/pattern_class_observations.csv   per (pattern, L) observables + kernel moments
    data/pattern_class_summary.csv        per pattern: s_min, ||X||, class

The analysis asks, in order:

  A. Does the cell-occupancy class (the project's existing axis) fix s_min?
  B. Do the cell COUNTS (n0,n1,n2) fix ||X||?   (if yes, counts suffice)
  C. Does the ARRANGEMENT of the defects fix ||X||?  We test the defect-dipole
     description: an empty cell is a - defect, a doubly occupied cell a + defect.
     Their displacement multiset (mod the supercell) is the cheapest arrangement
     invariant.
  D. Does the exact first-moment law  a_R = (2/m) M  hold for every measured point?

Usage:  python pattern_class_analysis.py
"""
import csv
import math
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
OBS = os.path.join(ROOT, "data", "pattern_class_observations.csv")
SUM = os.path.join(ROOT, "data", "pattern_class_summary.csv")


def load(path):
    if not os.path.isfile(path):
        return []
    return list(csv.DictReader(open(path, encoding="utf-8")))


def cell_occupancies(bits, m):
    """cells[i] for cell (x,y), i = (y-1)*m + (x-1); bits are A(x,y),B(x,y), x fastest."""
    out = []
    for y in range(1, m + 1):
        for x in range(1, m + 1):
            j = 2 * ((x - 1) + m * (y - 1))
            out.append(int(bits[j]) + int(bits[j + 1]))
    return out


def defect_structure(bits, m):
    """(-) empty cells, (+) double cells, and the displacement multiset between them."""
    cells = cell_occupancies(bits, m)
    minus = [i for i, v in enumerate(cells) if v == 0]
    plus = [i for i, v in enumerate(cells) if v == 2]
    disps = []
    for a in minus:
        for b in plus:
            ax, ay = a % m, a // m
            bx, by = b % m, b // m
            disps.append(((bx - ax) % m, (by - ay) % m))
    # dipole: sum of (plus - minus) positions, mod m
    dip = [0, 0]
    for i in plus:
        dip[0] += i % m
        dip[1] += i // m
    for i in minus:
        dip[0] -= i % m
        dip[1] -= i // m
    return minus, plus, sorted(disps), (dip[0] % m, dip[1] % m)


# --------------------------------------------------------------- symmetries
def bit_index(m, x, y, s):
    """Bit index of sublattice s (0=A, 1=B) in cell (x,y), 1-based x,y."""
    return 2 * ((x - 1) + m * (y - 1)) + s


def apply_map(bits, m, src):
    """Rebuild the bit string with new(x,y,s) = old(src(x,y))."""
    out = ["0"] * len(bits)
    for y in range(1, m + 1):
        for x in range(1, m + 1):
            ax, ay = src(x, y)
            for s in (0, 1):
                out[bit_index(m, x, y, s)] = bits[bit_index(m, ax, ay, s)]
    return "".join(out)


def translate(bits, m, sx, sy):
    return apply_map(bits, m, lambda x, y: ((x - sx - 1) % m + 1, (y - sy - 1) % m + 1))


def rotate90(bits, m):
    """90 degree rotation about the lattice site (1,1).

    Rotating about the supercell CENTRE instead would map (x,y) -> (m+1-y, x), which
    flips the checkerboard parity x+y for even m and therefore swaps the two
    sublattices - that is not a symmetry of the model.  Rotating about a site,
    (a,b) -> (b, 2-a), keeps x+y mod 2 and so maps A->A, B->B.

    NOTE: this is NOT a symmetry of the Hamiltonian.  The lattice is a brick-wall
    (honeycomb) embedding: A(x,y) couples to B(x,y), B(x+1,y) and B(x,y-1), a bond
    set that no 90 degree rotation preserves.  It is kept only so the analysis can
    *show* that it fails (264 of 512 m=3 patterns change ||X||).
    """
    return apply_map(bits, m, lambda x, y: ((2 - y - 1) % m + 1, (x - 1) % m + 1))


def rotate180(bits, m):
    """180 degree rotation - a genuine symmetry of the brick-wall bond set."""
    return apply_map(bits, m, lambda x, y: ((2 - x - 1) % m + 1, (2 - y - 1) % m + 1))


def reflect_diag(bits, m):
    """x <-> y reflection - a genuine symmetry of the brick-wall bond set."""
    return apply_map(bits, m, lambda x, y: (y, x))


def reflect(bits, m):
    """x -> 2-x reflection.  Kept to demonstrate that it FAILS (the bond set has a
    preferred x direction), unlike reflect_diag."""
    return apply_map(bits, m, lambda x, y: ((2 - x - 1) % m + 1, y))


def complement(bits, m):
    """Swap the fermion between the two sublattices in every cell: E <-> F."""
    out = []
    for y in range(1, m + 1):
        for x in range(1, m + 1):
            a = bits[bit_index(m, x, y, 0)]
            b = bits[bit_index(m, x, y, 1)]
            out.append(b + a)
    return "".join(out)


def orbit_key(bits, m, with_complement=True):
    """Canonical representative of the orbit under the model's TRUE symmetry group:

        translations x {1, rot180, ref_diag, ref_diag*rot180} x (optionally complement)

    rot90, ref_x and ref_y are deliberately excluded: they are not symmetries of the
    brick-wall bond set and 264/512 m=3 patterns change ||X|| under them.
    """
    images = []
    base = [bits, rotate180(bits, m), reflect_diag(bits, m),
            reflect_diag(rotate180(bits, m), m)]
    for b in base:
        for sy in range(m):
            for sx in range(m):
                images.append(translate(b, m, sx, sy))
    if with_complement:
        for im in list(images):
            images.append(complement(im, m))
    return min(images)


def main():
    obs, summary = load(OBS), load(SUM)
    if not summary:
        print("no data; run pattern_classes.jl first")
        return

    # ---- per-pattern table -------------------------------------------------
    pats = {}
    for s in summary:
        m = int(s["m"])
        bits = s["label"].split("_", 1)[1]
        obstructed = s["obstructed"].strip().lower() == "true"
        rec = dict(m=m, label=s["label"], cls=int(s["cls"]),
                   n0=int(s["n0"]), n1=int(s["n1"]), n2=int(s["n2"]),
                   obstructed=obstructed,
                   smin=float(s["smin"]), xnorm=float(s["xnorm"]))
        if not obstructed:
            minus, plus, disps, dip = defect_structure(bits, m)
            rec.update(minus=minus, plus=plus, disps=disps, dipole=dip)
        pats[s["label"]] = rec

    # ---- D. exact law ------------------------------------------------------
    print("=" * 78)
    print("D. exact first-moment law   a_R = (4/m^2) * M")
    by_m = defaultdict(list)
    for r in obs:
        m = int(r["m"])
        momx = float(r["momx"])
        if momx <= 0:
            continue
        by_m[m].append(abs(float(r["A_A"]) / int(r["L"]) / ((4.0 / m ** 2) * momx) - 1.0))
    for m in sorted(by_m):
        v = by_m[m]
        print("   m=%d: %4d points, max deviation %.2e, mean %.2e"
              % (m, len(v), max(v), sum(v) / len(v)))

    # ---- A. does the occupancy class fix s_min? ---------------------------
    print()
    print("=" * 78)
    print("A. does the occupancy class fix s_min?")
    for m in sorted({p["m"] for p in pats.values()}):
        for cls in (1, 2, 3):
            sub = [p for p in pats.values() if p["m"] == m and p["cls"] == cls
                   and not p["obstructed"]]
            if not sub:
                continue
            v = [p["smin"] for p in sub]
            print("   m=%d class %d: n=%4d  s_min %.5f .. %.5f   ratio %7.1fx   "
                  "||X|| %.3f .. %.3f"
                  % (m, cls, len(sub), min(v), max(v), max(v) / min(v),
                     min(p["xnorm"] for p in sub), max(p["xnorm"] for p in sub)))

    # ---- B. do the counts fix ||X||? --------------------------------------
    print()
    print("=" * 78)
    print("B. do the cell counts (n0,n1,n2) fix ||X||?")
    groups = defaultdict(list)
    for p in pats.values():
        if p["obstructed"]:
            continue
        groups[(p["m"], p["n0"], p["n1"], p["n2"])].append(p)
    bad = 0
    for key in sorted(groups):
        sub = groups[key]
        if len(sub) < 2:
            continue
        vals = sorted({round(p["xnorm"], 4) for p in sub})
        if len(vals) > 1:
            bad += 1
            print("   m=%d counts %s: n=%3d but %d distinct ||X||: %s"
                  % (key[0], key[1:], len(sub), len(vals), vals[:4]))
    print("   -> %d count-groups split across several ||X|| values "
          "(counts do NOT determine the measurement strength)" % bad)

    # ---- C. does the defect arrangement fix ||X||? ------------------------
    print()
    print("=" * 78)
    print("C. does the defect arrangement fix ||X||?")
    print("   grouping by (m, displacement multiset between - and + defects)")
    agroups = defaultdict(list)
    for p in pats.values():
        if p["obstructed"] or (not p["minus"] and not p["plus"]):
            continue
        agroups[(p["m"], tuple(p["disps"]))].append(p)
    split = 0
    for key in sorted(agroups):
        sub = agroups[key]
        vals = sorted({round(p["xnorm"], 4) for p in sub})
        if len(vals) > 1:
            split += 1
            print("   m=%d disps %s: n=%3d -> %d distinct ||X|| %s"
                  % (key[0], key[1], len(sub), len(vals), vals[:4]))
    print("   -> %d arrangement-groups split (0 means the displacement multiset "
          "fixes ||X|| completely)" % split)

    # ---- summary of the ||X|| spectrum ------------------------------------
    print()
    print("=" * 78)
    print("||X|| spectrum and how many patterns sit at each value")
    spec = defaultdict(lambda: defaultdict(int))
    for p in pats.values():
        if p["obstructed"]:
            continue
        spec[p["m"]][round(p["xnorm"], 3)] += 1
    for m in sorted(spec):
        vals = spec[m]
        print("   m=%d: %d distinct values" % (m, len(vals)))
        for xn in sorted(vals):
            print("        ||X|| = %10.3f   %4d patterns" % (xn, vals[xn]))

    # ---- law table for the report ----------------------------------------
    print()
    print("=" * 78)
    print("per-pattern entropy influence at the largest L")
    lmax = max(int(r["L"]) for r in obs) if obs else 0
    rows = [r for r in obs if int(r["L"]) == lmax]
    print("   L = %d, %d patterns" % (lmax, len(rows)))
    tbl = defaultdict(list)
    for r in rows:
        tbl[(int(r["m"]), int(r["cls"]))].append(float(r["A_A"]) / lmax)
    for key in sorted(tbl):
        v = sorted(tbl[key])
        print("   m=%d class %d: n=%4d  a_R %.4f .. %.4f"
              % (key[0], key[1], len(v), v[0], v[-1]))


    # ---- E. symmetry structure -------------------------------------------
    print()
    print("=" * 78)
    print("E. symmetry: what the strength is invariant under")
    # per-pattern a_R at the largest L available for that m
    lmax_of_m = {}
    for r in obs:
        m, L = int(r["m"]), int(r["L"])
        lmax_of_m[m] = max(lmax_of_m.get(m, 0), L)
    aR_of, aC_of = {}, {}
    for r in obs:
        m, L = int(r["m"]), int(r["L"])
        if L != lmax_of_m[m]:
            continue
        aR_of[r["label"]] = float(r["A_A"]) / L
        aC_of[r["label"]] = float(r["A_C"]) / L

    by_bits = {}
    for lab, p in pats.items():
        if p["obstructed"]:
            continue
        bits = lab.split("_", 1)[1]
        by_bits[(p["m"], bits)] = lab

    checks = defaultdict(lambda: [0, 0.0])   # name -> [n, worst relative change]

    def cmp(name, lab_a, lab_b, va, vb, scale):
        if lab_a is None or lab_b is None or va is None or vb is None:
            return
        checks[name][0] += 1
        if scale > 0:
            checks[name][1] = max(checks[name][1], abs(va - vb) / scale)

    for (m, bits), lab in by_bits.items():
        xn = pats[lab]["xnorm"]
        aR = aR_of.get(lab)
        # translations must leave ||X|| invariant (X(k) picks up phases that cancel
        # in the ratio only for ||X||, not for X itself)
        for sx, sy in ((1, 0), (0, 1)):
            tb = translate(bits, m, sx, sy)
            other = by_bits.get((m, tb))
            if other is None:
                continue
            cmp("translation: ||X||", lab, other, xn, pats[other]["xnorm"], max(xn, 1e-30))
            cmp("translation: a_R (NOT expected invariant)", lab, other, aR,
                aR_of.get(other), max(abs(aR or 0), 1e-30))
        # the point-group operations, split into genuine symmetries and the ones the
        # brick-wall bond set breaks
        for name, f, genuine in (
            ("rot180", rotate180, True),
            ("ref_diag", reflect_diag, True),
            ("rot90", rotate90, False),
            ("ref_x", reflect, False),
        ):
            ob = f(bits, m)
            other = by_bits.get((m, ob))
            if other is None:
                continue
            tag = "" if genuine else " (expected to FAIL)"
            cmp("%s: ||X||%s" % (name, tag), lab, other, xn,
                pats[other]["xnorm"], max(xn, 1e-30))
        # complement swaps E and F, so X -> X^{-1}.  sigma_max(X^{-1}) = 1/sigma_min(X),
        # which equals sigma_max(X) only if sigma_min*sigma_max = 1; empirically it does,
        # so ||X|| survives while a_R does not.
        cb = complement(bits, m)
        other = by_bits.get((m, cb))
        if other is not None:
            cmp("complement: ||X||", lab, other, xn, pats[other]["xnorm"], max(xn, 1e-30))
            cmp("complement: a_R", lab, other, aR, aR_of.get(other), max(abs(aR or 0), 1e-30))

    for name in sorted(checks):
        n, worst = checks[name]
        print("   %-46s n=%5d  worst rel. change %.2e" % (name, n, worst))

    # orbit decomposition
    print()
    print("   orbit decomposition (translations x D4, with and without E<->F)")
    for m in sorted({p["m"] for p in pats.values()}):
        sub = [lab for lab, p in pats.items()
               if p["m"] == m and not p["obstructed"]]
        for wc in (False, True):
            orbs = defaultdict(list)
            for lab in sub:
                bits = lab.split("_", 1)[1]
                orbs[orbit_key(bits, m, wc)].append(lab)
            nxn = len({round(pats[l]["xnorm"], 6) for l in sub})
            print("      m=%d complement=%-5s: %4d patterns -> %4d orbits "
                  "(%d distinct ||X||)"
                  % (m, wc, len(sub), len(orbs), nxn))
            if not wc:
                # is ||X|| constant on each orbit?  Use a relative tolerance: the
                # most ill-conditioned patterns have ||X|| ~ 4000 from s_min ~ 2e-4,
                # so absolute rounding at 1e-6 would report false splits.
                bad = 0
                for v in orbs.values():
                    vals = [pats[l]["xnorm"] for l in v]
                    lo, hi = min(vals), max(vals)
                    if hi - lo > 1e-6 * hi:
                        bad += 1
                print("         orbits splitting ||X||: %d" % bad)


if __name__ == "__main__":
    main()
