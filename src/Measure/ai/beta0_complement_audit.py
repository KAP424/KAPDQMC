"""Exact, fixed-origin occupancy complement checks; no entropy calculation.

Only an unshifted site-by-site complement invokes the already established
beta=0 chiral particle-hole entropy identity. Translation relations are listed
separately and do not assert equality for the same fixed region cuts.
"""
import hashlib
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_beta0_study"


def main():
    paths = [OUT/"baseline_provenance.json", OUT/"study_design.json", OUT/"confirmation_plan.json"]
    raw = [p.read_bytes() for p in paths]
    provenance, design, confirmation = [json.loads(b) for b in raw]
    names = design["patterns"] + ["uniform_B"]
    assert len(names) == len(set(names)) == 17
    selected = set(confirmation["patterns"])
    pilot = set(design["patterns"]) - selected
    metadata = provenance["patterns"]
    patterns = []
    for name in names:
        p = metadata[name]
        bits = p["occupied_bits"]
        m = p["m"]
        assert len(bits) == 2*m*m and set(bits) <= {"0", "1"}
        assert bits.count("1") == m*m
        patterns.append(dict(name=name, m=m, bits=bits,
                             complement_bits="".join("1" if b=="0" else "0" for b in bits),
                             cohort="confirmation" if name in selected else "pilot_only" if name in pilot else "analytic_complement_control"))

    def at(p, x, y, a):
        return int(p["bits"][2*((x%p["m"])+(y%p["m"])*p["m"])+a])

    def equal(p, q, dx=0, dy=0, complement=False):
        period = math.lcm(p["m"], q["m"])
        return all(at(p,x,y,a) == (1-at(q,x+dx,y+dy,a) if complement else at(q,x+dx,y+dy,a))
                   for y in range(period) for x in range(period) for a in (0,1))

    exact_duplicates, exact_complements, translated, self_translated_complements = [], [], [], []
    for p, q in itertools.combinations(patterns, 2):
        M = math.lcm(p["m"],q["m"])
        base = dict(pattern_a=p["name"],pattern_b=q["name"],comparison_period=M)
        if equal(p,q):
            exact_duplicates.append(base)
        if equal(p,q,complement=True):
            exact_complements.append(base)
        for is_complement in (False,True):
            if equal(p,q,complement=is_complement):
                continue
            shifts = [[dx,dy] for dx in range(M) for dy in range(M)
                      if (dx or dy) and equal(p,q,dx,dy,is_complement)]
            if shifts:
                translated.append(dict(**base,relation="translated_complement" if is_complement else "translated_duplicate",
                                       shifts=shifts,same_fixed_cut_entropy_equality_proved=False))
    for p in patterns:
        shifts = [[dx,dy] for dx in range(p["m"]) for dy in range(p["m"])
                  if (dx or dy) and equal(p,p,dx,dy,True)]
        if shifts:
            self_translated_complements.append(dict(pattern=p["name"],shifts=shifts,same_fixed_cut_entropy_equality_proved=False))
    pilot_matches = {name:[r["pattern_b"] if r["pattern_a"]==name else r["pattern_a"]
                          for r in exact_complements
                          if (r["pattern_a"]==name and r["pattern_b"] in selected) or
                             (r["pattern_b"]==name and r["pattern_a"] in selected)] for name in sorted(pilot)}
    files = {p.relative_to(ROOT).as_posix():hashlib.sha256(b).hexdigest() for p,b in zip(paths,raw)}
    files[Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result = dict(status="PASS",generated_utc=datetime.now(timezone.utc).isoformat(),
                  scope="Exact occupation bits at a fixed site origin; no fits or entropy values used",
                  indexing="bits[2*(x+y*m)+a], x,y zero-based cell coordinates, a=0 A / a=1 B; compare different m on lcm(m1,m2)",
                  checked_patterns=17,checked_distinct_pairs=136,patterns=patterns,
                  fixed_origin_duplicates=exact_duplicates,fixed_origin_complements=exact_complements,
                  pilot_only_to_confirmation_complement=pilot_matches,
                  translated_relations_not_used_to_infer_same_cut_entropy=translated,
                  self_translated_complements_not_used_to_infer_same_cut_entropy=self_translated_complements,
                  proven_identity="At beta=0, C_Pc(theta)=I-Gamma C_P(theta) Gamma. Gamma is diagonal by site, so each fixed-region spectrum maps lambda to 1-lambda and S2 is unchanged.",
                  limitations="Translation relations alone do not prove equal D for the same anchored cuts. No rotations, sublattice exchanges, or arbitrary orbital permutations were admitted as fixed-site complements.",
                  files=files)
    for p,b in zip(paths,raw):
        assert p.read_bytes()==b,"Input changed during audit"
    target=OUT/"complement_audit.json"
    target.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:result[k] for k in ("status","fixed_origin_duplicates","fixed_origin_complements","pilot_only_to_confirmation_complement",
                                          "translated_relations_not_used_to_infer_same_cut_entropy","self_translated_complements_not_used_to_infer_same_cut_entropy")},indent=2))


if __name__ == "__main__":
    main()
