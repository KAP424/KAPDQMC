"""Freeze the beta=0 pilot grid; never select points by equilibrium agreement."""
from pathlib import Path
from datetime import datetime, timezone
import csv, hashlib, json, math

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_beta0_study"
NAMES = [
    "uniform_A", "one_B_2", "one_B_3", "two_B_3", "diagonal_B_3",
    "special_matched_a", "special_matched_b", "general_matched_a",
    "general_matched_b", "random_half_3", "full_empty_cluster3",
    "near02_irregular3", "pure02_block4", "pure02_irregular4",
    "pure02_staircase4", "pure02_notched_stripe4",
]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    plan_path = OUT / "study_design.json"
    if plan_path.exists():
        raise SystemExit("Frozen study design already exists; do not overwrite it.")
    base = OUT / "baseline_observations.csv"
    provenance = OUT / "baseline_provenance.json"
    rows = list(csv.DictReader(base.open(encoding="utf-8-sig", newline="")))
    known = {(r["canonical_pattern"], int(r["L"]), float(r["theta"])): r for r in rows}
    targets = [(p,L,t,"pilot_time") for L in (72,108) for p in NAMES
               for t in [1+.5*i for i in range(19)]]
    targets += [(p,L,math.inf,"equilibrium_extension") for L in (180,216) for p in NAMES]
    jobs, reused = [], []
    for p,L,t,stage in targets:
        key = (p,L,t)
        record = dict(pattern=p,L=L,theta="Inf" if math.isinf(t) else f"{t:g}",beta=0,stage=stage)
        if key in known:
            row=known[key]
            assert row["ground_reachable"].lower()=="true"
            reused.append(record | {"source_id":row["source_id"]})
        else:
            jobs.append(record)
    for filename, data, fields in (("pilot_jobs.csv",jobs,["pattern","L","theta","beta","stage"]),
                                   ("pilot_reused.csv",reused,["pattern","L","theta","beta","stage","source_id"])):
        with (OUT/filename).open("w",encoding="utf-8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(data)
    plan = dict(
        recorded_utc=datetime.now(timezone.utc).isoformat(),
        status="Frozen exploratory grid before new entropy computation",
        question="At beta=0 which initial configurations have independently stable time scaling consistent with equilibrium?",
        patterns=NAMES, beta=0, total_target_keys=len(targets), new_jobs=len(jobs), reused_keys=len(reused),
        pilot_sizes=[72,108], pilot_theta=[1+.5*i for i in range(19)], equilibrium_extension_sizes=[180,216],
        protocol="Same spinless HoneyComb120, half filling, PBC, natural-log S2, fixed cut origin; D=(2SA+SC)/3-SB. Exact initial spectral support.",
        exclusions={"uniform_B":"Complement of uniform_A: exact beta=0 regional entropy equivalence, independently checked.",
                    "five_blocked_m2":"Existing data retained as controls; not ground-state coefficient candidates."},
        analysis_rules=[
            "The pilot maps initial transients and finite-size sensitivity; it is not a certification grid.",
            "Do not fit points merely because they agree with an equilibrium coefficient.",
            "Use actual same-theta size plateaus and raw adjacent log slopes before interpreting corrected fits.",
            "Final time fits must satisfy theta>=4 and theta/L<=0.05; early and later pilot points remain diagnostics.",
            "Evaluate lower-window, upper-window and inverse-correction-order sensitivity, not a single narrow fit.",
            "Report fit-sensitivity ranges as such, never as statistical confidence intervals.",
            "Any selected confirmation grid will be recorded separately, after the exploratory pilot but before confirmation computations.",
            "Compare with each pattern's equilibrium estimate only after time-side eligibility is evaluated.",
            "Finite-window failure is unresolved evidence, not proof that the asymptotic coefficient differs.",
        ],
        sources={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in (base,provenance,Path(__file__),OUT/"pilot_jobs.csv",OUT/"pilot_reused.csv")},
        reference={"url":"https://www.nature.com/articles/s41467-026-75645-x",
                   "scope":"AFM product-state benchmark; fixed-time size plateau followed by logarithmic time scaling. This does not establish arbitrary-pattern universality."},
    )
    plan_path.write_text(json.dumps(plan,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:plan[k] for k in ("total_target_keys","new_jobs","reused_keys")},indent=2))

if __name__=="__main__": main()
