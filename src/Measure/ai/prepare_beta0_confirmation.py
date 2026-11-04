"""Record a separate confirmation grid after the exploratory time pilot."""
from pathlib import Path
from datetime import datetime, timezone
import csv, hashlib, json, math

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"analysis_beta0_study"
SELECTED=["uniform_A","one_B_2","diagonal_B_3","random_half_3",
          "one_B_3","special_matched_a","general_matched_a","near02_irregular3",
          "pure02_block4","pure02_notched_stripe4"]

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def write_csv(path,rows,fields):
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
    destination=OUT/"confirmation_plan.json"
    if destination.exists(): raise SystemExit("Confirmation plan already frozen")
    study=json.loads((OUT/"study_design.json").read_text())
    baseline=OUT/"baseline_observations.csv"
    raw=ROOT/"data/beta0_study_observations.csv"
    data={}
    for path in (baseline,raw):
        content=path.read_bytes()
        if not content.endswith(b"\n"): raise ValueError("Wait for complete CSV row")
        rows=list(csv.DictReader(content.decode("utf-8-sig").splitlines()))
        for n,r in enumerate(rows,2):
            if float(r["beta"])!=0:continue
            name=r.get("canonical_pattern",r["pattern"])
            key=(name,int(r["L"]),float(r["theta"]))
            record={"pattern":name,"L":key[1],"theta":key[2],
                    **{f:float(r[f]) for f in ("S_A","S_C","S_B","D")},
                    "source":path.relative_to(ROOT).as_posix(),"source_row":n,
                    "source_row_sha256":hashlib.sha256(content.decode("utf-8-sig").splitlines()[n-1].encode()).hexdigest()}
            if key in data:
                assert max(abs(record[f]-data[key][f]) for f in ("S_A","S_C","S_B","D"))<1e-8
            else:data[key]=record
    evidence=[]
    for name in study["patterns"]:
        for size in (72,108):
            for theta in study["pilot_theta"]:
                evidence.append(data[name,size,theta])
    assert len(evidence)==608
    evidence_file=OUT/"confirmation_pilot_time_evidence.csv"
    write_csv(evidence_file,evidence,list(evidence[0]))
    targets=[dict(pattern=p,L=L,theta=f"{t:g}",beta=0,stage="confirmation_time")
             for L,lo,hi in ((144,4,7),(180,4,9),(216,4,10))
             for p in SELECTED for t in [lo+.5*i for i in range(round(2*(hi-lo))+1)]]
    jobs=[];reuse=[]
    for r in targets:
        key=(r["pattern"],r["L"],float(r["theta"]))
        (reuse if key in data else jobs).append(r)
    fields=["pattern","L","theta","beta","stage"]
    write_csv(OUT/"confirmation_jobs.csv",jobs,fields)
    write_csv(OUT/"confirmation_reused.csv",reuse,fields)
    plan=dict(
        recorded_utc=datetime.now(timezone.utc).isoformat(),
        status="Frozen after complete exploratory time pilot and before confirmation calculations",
        patterns=SELECTED,beta=0,total_target_keys=len(targets),new_jobs=len(jobs),reused_keys=len(reuse),
        primary=dict(L=216,comparison_L=180,theta_min=4,theta_max=9,theta_step=.5,model="log_inv1_inv2"),
        third_size=dict(sizes=[144,180,216],theta_min=4,theta_max=7,theta_step=.5),
        additional_window=dict(L=216,theta_min=5,theta_max=10,theta_step=.5),
        model_orders=[1,2,3],order3_minimum_points=11,
        windows=[[4,7],[4,8],[4,9],[4.5,9],[4,8.5],[5,9],[5,10]],
        thresholds=dict(theta_min=4,theta_over_L_max=.05,size_coefficient_drift=.02,
                        D_difference_over_fitted_log_span=.01,model_drift=.05,window_drift=.05,
                        equilibrium_sensitivity=.05,postselection_relative_eq_time_gap=.05),
        comparison_rule="Time-side eligibility is evaluated without equilibrium. Numerical agreement requires time-side checks, stable equilibrium fits, <=5% relative endpoint difference and overlapping model/window sensitivity ranges; ranges are not confidence intervals. This is finite-window evidence, not an equality theorem.",
        selection_method="Exploratory coverage and boundary-overlap diagnostics, without selecting for equilibrium agreement. Existing dense large-size baselines plus cross-class new checks and two difficult class-2 controls; not an exhaustive automatic ranking.",
        selection_reasons={
            "uniform_A":"Analytic and literature benchmark; fill missing half-step points.",
            "one_B_2":"Existing multi-size benchmark; test broader window and late shifts.",
            "diagonal_B_3":"Existing multi-size data; retain imperfect model stability as a test.",
            "random_half_3":"Existing multi-size mixed-class data; retain smaller overlap as a test.",
            "one_B_3":"Strong regular boundary overlap; existing time coverage stops at L108.",
            "special_matched_a":"Regular boundary overlap and relatively flat raw slopes; missing large sizes.",
            "general_matched_a":"Mixed class with regular low-energy overlap and small same-time size differences.",
            "near02_irregular3":"Mostly empty/double mixed example; raw slopes are smooth but narrow-model ambiguity remains.",
            "pure02_block4":"Class-2 control with regular overlap but unresolved previous corrected fits.",
            "pure02_notched_stripe4":"Class-2 control with small gapped overlap; raw size agreement alone was insufficient.",
        },
        not_confirmed=[p for p in study["patterns"] if p not in SELECTED],
        not_confirmed_interpretation="Pilot-only coverage is insufficient for a new confirmation label; do not call these patterns impossible or failed asymptotically.",
        sources={p.relative_to(ROOT).as_posix():sha(p) for p in
                 (OUT/"study_design.json",baseline,OUT/"baseline_provenance.json",evidence_file,
                  OUT/"inventory.csv",OUT/"physics_summary.json",OUT/"confirmation_jobs.csv",
                  OUT/"confirmation_reused.csv",Path(__file__))},
    )
    destination.write_text(json.dumps(plan,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:plan[k] for k in ("total_target_keys","new_jobs","reused_keys")},indent=2))

if __name__=="__main__":main()
