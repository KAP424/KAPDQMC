"""Independent final beta=0 fit audit (80 decimal digit rectangular QR).

Run only after the final analysis and final data validation have completed.
The reference computations do not call the analysis fit/selection helpers.
--self-test checks the QR implementation on synthetic data and writes no receipt.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import itertools
import json
import math
from pathlib import Path
import sys
from collections import defaultdict
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_beta0_study"
WINDOWS = ((4., 7.), (4., 8.), (4., 9.), (4.5, 9.), (4., 8.5), (5., 9.), (5., 10.))
SIZES = (72, 108, 144, 180, 216)
MODELS = {1: "log_inv1", 2: "log_inv1_inv2", 3: "log_inv1_inv2_inv3"}
ATOL, RTOL = 2e-8, 2e-8


def load_mp():
    try:
        import mpmath
    except ImportError:
        sys.path.insert(0, str(ROOT / "analysis_beta_patterns/audit_dependencies"))
        import mpmath
    mpmath.mp.dps = 80
    return mpmath


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class Audit:
    def __init__(self):
        self.files = {}
        self.checks = 0
        self.errors = defaultdict(float)
        self.worst = {}

    def read(self, path):
        path = Path(path)
        if not path.is_absolute():
            path = ROOT / path
        raw = path.read_bytes()
        key = path.relative_to(ROOT).as_posix()
        checksum = digest(raw)
        if key in self.files:
            assert self.files[key] == checksum, ("File changed during audit", key)
        self.files[key] = checksum
        return raw

    def js(self, path):
        return json.loads(self.read(path))

    def csv(self, name):
        return list(csv.DictReader(io.StringIO(self.read(OUT / name).decode("utf-8-sig"))))

    def exact(self, got, want, where):
        self.checks += 1
        assert got == want, (where, got, want)

    def number(self, got, want, where, family="derived", atol=ATOL, rtol=RTOL):
        self.checks += 1
        if want is None:
            assert got in (None, ""), (where, got, want)
            return
        a, b = float(got), float(want)
        assert math.isfinite(a) and math.isfinite(b), (where, got, want)
        error = abs(a-b)
        if error > self.errors[family]:
            self.errors[family] = error
            self.worst[family] = dict(where=str(where), reported=a, reference=b)
        assert error <= atol + rtol*abs(b), (where, got, want, error, atol+rtol*abs(b))

    def unchanged(self):
        for name, sha in self.files.items():
            assert digest((ROOT/name).read_bytes()) == sha, ("Changed input", name)


def truth(x):
    assert x in (True, False, "True", "False", "true", "false"), x
    return x is True or x in ("True", "true")


def grid(values):
    return ",".join(f"{float(v):g}" for v in sorted(values))


def required(lo, hi):
    return [i/2 for i in range(round(2*lo), round(2*hi)+1)]


def ratio(a, b):
    return abs(a)/abs(b) if b is not None and abs(b)>1e-15 else None


def keyed(rows, columns):
    result = {}
    for row in rows:
        key = tuple(row[k] for k in columns)
        assert key not in result, ("Duplicate output key", key)
        result[key] = row
    return result


class ReferenceFits:
    def __init__(self, mp):
        self.mp = mp
        self.cache = {}

    def fit(self, points, axis, order):
        points = sorted(points, key=lambda p: p[axis])
        if len(points) < order+4:
            return None
        key = (order, tuple((str(p[axis]), p["D_decimal"]) for p in points))
        if key in self.cache:
            return self.cache[key]
        mp = self.mp
        x = [mp.mpf(str(p[axis])) for p in points]
        y = mp.matrix([mp.mpf(p["D_decimal"]) for p in points])
        A = mp.matrix([[mp.log(v), mp.mpf(1)]+[v**(-j) for j in range(1,order+1)] for v in x])
        coefficient, _ = mp.qr_solve(A, y)
        residual = A*coefficient-y
        correction = [sum(A[i,j]*coefficient[j] for j in range(2,order+2)) for i in range(len(x))]
        span = abs(coefficient[0]*mp.log(x[-1]/x[0]))
        result = dict(model=MODELS[order], inverse_order=order, n=len(x), parameters=order+2,
                      residual_dof=len(x)-order-2, x_min=float(x[0]), x_max=float(x[-1]), grid=grid(x),
                      c_fit=float(coefficient[0]), constant=float(coefficient[1]),
                      inv1=float(coefficient[2]), inv2=float(coefficient[3]) if order>=2 else 0.,
                      inv3=float(coefficient[4]) if order>=3 else 0.,
                      rmse=float(mp.sqrt(sum(v*v for v in residual)/len(x))),
                      max_abs_residual=float(max(abs(v) for v in residual)),
                      raw_D_span=float(max(y)-min(y)),
                      raw_endpoint_secant=float((y[len(x)-1]-y[0])/mp.log(x[-1]/x[0])),
                      correction_span=float(max(correction)-min(correction)),
                      correction_to_fitted_log_span=float((max(correction)-min(correction))/span) if span else None)
        self.cache[key] = result
        return result


def check_fields(audit, row, expected, label):
    for key, want in expected.items():
        got = row[key]
        if isinstance(want, bool):
            audit.exact(truth(got), want, (label,key))
        elif isinstance(want, str):
            audit.exact(got, want, (label,key))
        else:
            family = "c_fit" if key=="c_fit" else "fit_parameters" if key in ("constant","inv1","inv2","inv3") else "derived"
            audit.number(got, want, (label,key), family)


def reconstruct(audit, fitter, observations, names, metadata):
    times, limits = defaultdict(dict), defaultdict(list)
    for p in observations:
        if p["pattern"] not in names:
            continue
        if math.isinf(p["theta"]):
            limits[p["pattern"]].append(p)
        else:
            assert p["theta"] not in times[p["pattern"],p["L"]]
            times[p["pattern"],p["L"]][p["theta"]] = p
    coverage, tf, ef, pairs, differences = [], [], [], [], []
    for name in names:
        for L in SIZES:
            data = times[name,L]
            for lo,hi in WINDOWS:
                target = required(lo,hi)
                present = [t for t in target if t in data]
                missing = [t for t in target if t not in data]
                window = f"{lo:g}-{hi:g}"
                coverage.append(dict(pattern=name,L=L,window=window,required_grid=grid(target),present_grid=grid(present),
                                     missing_grid=grid(missing),complete=not missing,n=len(present),theta_over_L_max=hi/L,
                                     scale_screen=hi/L<=.05,all_actual_theta=grid(data)))
                chosen = [data[t] for t in present]
                for order in (1,2,3):
                    fit = fitter.fit(chosen,"theta",order)
                    if fit:
                        tf.append(dict(pattern=name,L=L,window=window,requested_min=lo,requested_max=hi,
                                       complete_grid=not missing,formal_robustness_eligible=not missing and (order<3 or len(chosen)>=11),
                                       scale_screen=hi/L<=.05,warning="incomplete_grid" if missing else
                                       "five_parameter_fit_has_fewer_than_11_points" if order==3 and len(chosen)<11 else "",**fit))
        cuts = (36,48,72) if metadata[name]["m"]==4 else (36,54,72)
        for lo,hi,order in itertools.product(cuts,(144,216),(1,2)):
            chosen = [p for p in limits[name] if lo<=p["L"]<=hi]
            fit = fitter.fit(chosen,"L",order)
            if fit:
                ef.append(dict(pattern=name,requested_min=lo,requested_max=hi,has_requested_upper=any(p["L"]==hi for p in chosen),
                               state_role="equilibrium_ground",**fit))
        for small,large in itertools.combinations(SIZES,2):
            left,right=times[name,small],times[name,large]
            for theta in sorted(left.keys() & right.keys()):
                differences.append(dict(pattern=name,L_small=small,L_large=large,theta=theta,D_small=left[theta]["D"],
                                        D_large=right[theta]["D"],D_difference=right[theta]["D"]-left[theta]["D"]))
            for lo,hi in WINDOWS:
                target=required(lo,hi)
                common=[t for t in target if t in left and t in right]
                if len(common)<2:
                    continue
                a,b=[left[t] for t in common],[right[t] for t in common]
                delta=[y["D"]-x["D"] for x,y in zip(a,b)]
                values=[p["D"] for p in b]
                span=max(values)-min(values)
                logrange=math.log(common[-1]/common[0])
                rec=dict(pattern=name,L_small=small,L_large=large,window=f"{lo:g}-{hi:g}",grid=grid(common),n=len(common),
                         complete_grid=common==target,scale_screen_both=hi/small<=.05,raw_D_span_large=span,
                         max_abs_D_difference=max(map(abs,delta)),raw_size_difference_over_D_span=ratio(max(map(abs,delta)),span),
                         raw_change_of_size_difference_over_D_span=ratio(max(delta)-min(delta),span),
                         raw_endpoint_slope_small=(a[-1]["D"]-a[0]["D"])/logrange,
                         raw_endpoint_slope_large=(b[-1]["D"]-b[0]["D"])/logrange)
                for order in (1,2,3):
                    fa,fb=fitter.fit(a,"theta",order),fitter.fit(b,"theta",order)
                    if fa and fb:
                        ca,cb=fa["c_fit"],fb["c_fit"]
                        rec.update({f"order{order}_c_small":ca,f"order{order}_c_large":cb,
                                    f"order{order}_relative_size_drift":ratio(cb-ca,cb),
                                    f"order{order}_D_difference_to_fitted_log_span":ratio(max(map(abs,delta)),cb*logrange)})
                pairs.append(rec)
    for filename, expected, keys in (
        ("study_time_coverage.csv",coverage,("pattern","L","window")),
        ("study_time_fits.csv",tf,("pattern","L","window","inverse_order")),
        ("study_equilibrium_fits.csv",ef,("pattern","requested_min","requested_max","inverse_order")),
        ("study_size_comparisons.csv",pairs,("pattern","L_small","L_large","window")),
        ("study_same_theta_size_differences.csv",differences,("pattern","L_small","L_large","theta"))):
        reported=audit.csv(filename)
        audit.exact(len(reported),len(expected),(filename,"count"))
        # Sort numeric components canonically because CSV may spell an integer 4.0.
        def k(row):
            return tuple(str(row[field]) if field in ("pattern","window") else float(row[field]) for field in keys)
        index={k(row):row for row in reported}
        audit.exact(len(index),len(reported),(filename,"unique"))
        audit.exact(set(index),{k(row) for row in expected},(filename,"keys"))
        for row in expected:
            check_fields(audit,index[k(row)],row,(filename,k(row)))
    return times,limits,tf,ef,pairs


def audit_slopes(audit, observations, names, metadata, mp):
    groups=defaultdict(list)
    for p in observations:
        if p["pattern"] in names:
            if math.isinf(p["theta"]):
                groups[p["pattern"],"limit",0].append(p)
            elif p["theta"]>0:
                groups[p["pattern"],"time",p["L"]].append(p)
    expected=[]
    for (name,domain,L),points in sorted(groups.items()):
        axis="L" if domain=="limit" else "theta"
        points.sort(key=lambda p:p[axis])
        pairs=[("adjacent",a,b) for a,b in zip(points,points[1:])]
        pairs += [("doubling",a,b) for a,b in itertools.combinations(points,2) if b[axis]==2*a[axis]]
        for kind,a,b in pairs:
            x,y=mp.mpf(str(a[axis])),mp.mpf(str(b[axis]))
            slope=(mp.mpf(b["D_decimal"])-mp.mpf(a["D_decimal"]))/mp.log(y/x)
            expected.append(dict(pattern=name,domain=domain,L=L if domain=="time" else "",pair_kind=kind,
                                 x_low=float(x),x_high=float(y),x_geometric_mean=float(mp.sqrt(x*y)),
                                 D_low=a["D"],D_high=b["D"],log_slope=float(slope),
                                 theta_over_L_max=float(y)/L if domain=="time" else "",
                                 ground_reachable=True,occupancy_class=metadata[name]["occupancy_class"],
                                 limit_state="ground_state",parameter_meaning="finite_window_fit_not_certified_corner_coefficient",
                                 source_id_low=a["source_id"],source_id_high=b["source_id"]))
    reported=audit.csv("study_local_log_slopes.csv")
    def k(row):
        return row["pattern"],row["domain"],str(row["L"]),row["pair_kind"],float(row["x_low"]),float(row["x_high"])
    index={k(row):row for row in reported}
    audit.exact(len(index),len(reported),("raw slopes","unique"))
    audit.exact(set(index),{k(row) for row in expected},("raw slopes","keys"))
    for row in expected:
        check_fields(audit,index[k(row)],row,("raw slopes",k(row)))
    return expected


def choose_time(name, tf, pairs):
    """Reference selection has no equilibrium data or target in its arguments."""
    primary=next((r for r in tf if r["pattern"]==name and r["L"]==216 and r["window"]=="4-9" and r["inverse_order"]==2),None)
    pair=next((r for r in pairs if r["pattern"]==name and r["L_small"]==180 and r["L_large"]==216 and r["window"]=="4-9"),None)
    models=[r for r in tf if r["pattern"]==name and r["L"]==216 and r["window"]=="4-9" and r["formal_robustness_eligible"]]
    windows=[r for r in tf if r["pattern"]==name and r["L"]==216 and r["inverse_order"]==2 and r["complete_grid"] and r["scale_screen"]]
    c=primary["c_fit"] if primary else None
    model=max((ratio(r["c_fit"]-c,c) for r in models),default=None) if c else None
    window=max((ratio(r["c_fit"]-c,c) for r in windows),default=None) if c else None
    third=[next((r for r in pairs if r["pattern"]==name and r["L_small"]==a and r["L_large"]==b and r["window"]=="4-7"),None)
           for a,b in ((144,180),(180,216))]
    third_complete=all(r and r["complete_grid"] and "order2_relative_size_drift" in r for r in third)
    drifts=[r["order2_relative_size_drift"] for r in third] if third_complete else [None,None]
    decreasing=drifts[1]<drifts[0] if third_complete else None
    flags=dict(primary_complete=bool(primary and primary["complete_grid"] and primary["n"]>=11),
               pair_complete=bool(pair and pair["complete_grid"] and pair["scale_screen_both"]),
               models_complete=len(models)==3,windows_complete=len(windows)==7,third_complete=third_complete)
    missing=[k for k,v in flags.items() if not v]
    size=pair.get("order2_relative_size_drift") if pair else None
    raw=pair.get("order2_D_difference_to_fitted_log_span") if pair else None
    sign=bool(c and (any(r["c_fit"]*c<=0 for r in models+windows) or (pair and pair.get("order2_c_small",c)*c<=0)))
    passed=not missing and all(v is not None and v<=bound for v,bound in ((size,.02),(raw,.01),(model,.05),(window,.05)))
    passed=bool(passed and decreasing is True and drifts[1]<=.02 and not sign)
    return dict(primary_L=216,comparison_L=180,primary_window="4-9",primary_model="log_inv1_inv2",c_time_fit=c,
                time_sensitivity_min=min((r["c_fit"] for r in models+windows),default=None),
                time_sensitivity_max=max((r["c_fit"] for r in models+windows),default=None),**flags,
                size_drift=size,D_difference_to_fitted_log_span=raw,model_drift=model,window_drift=window,
                third_drift_144_180=drifts[0],third_drift_180_216=drifts[1],third_decreasing=decreasing,
                sign_changed=sign,missing_checks=";".join(missing),time_side_local_checks_pass=passed)


def audit_primary(audit,names,tf,ef,pairs,confirmation):
    reported=keyed(audit.csv("primary_coefficients.csv"),("pattern",))
    audit.exact(set(reported),{(n,) for n in names},("primary","names"))
    results=[]
    for name in names:
        expected=choose_time(name,tf,pairs)
        selected=name in confirmation
        passed=expected["time_side_local_checks_pass"]
        expected.update(confirmation_selected=selected,confirmation_local_checks_pass=selected and passed,
                        status="confirmed_local_checks_not_asymptotic_proof" if selected and passed else
                        "available_local_checks_candidate_not_confirmation_selected" if passed else
                        "unresolved_missing_confirmation_data" if expected["missing_checks"] else "unresolved_time_side_drift")
        eq=[r for r in ef if r["pattern"]==name and r["requested_max"]==216 and r["has_requested_upper"]]
        main=next((r for r in eq if r["requested_min"]==36 and r["inverse_order"]==1),None)
        c,ce=expected["c_time_fit"],main["c_fit"] if main else None
        eqvalues=[r["c_fit"] for r in eq]
        eqdrift=max((ratio(value-ce,ce) for value in eqvalues),default=None) if ce else None
        eqstable=len(eqvalues)==6 and eqdrift is not None and eqdrift<=.05
        gap=(c-ce)/abs(ce) if c is not None and ce is not None and ce else None
        tmin,tmax=expected["time_sensitivity_min"],expected["time_sensitivity_max"]
        overlap=max(min(eqvalues),tmin)<=min(max(eqvalues),tmax) if eqvalues and tmin is not None else None
        supported=selected and passed and eqstable and gap is not None and abs(gap)<=.05 and overlap is True
        expected.update(c_eq_fit=ce,eq_model="log_inv1",eq_requested_window="36-216",eq_actual_grid=main["grid"] if main else "unavailable",
                        eq_sensitivity_min=min((r["c_fit"] for r in eq),default=None),eq_sensitivity_max=max((r["c_fit"] for r in eq),default=None),
                        time_minus_eq=c-ce if c is not None and ce is not None else None,
                        relative_time_eq_gap=gap,equilibrium_sensitivity=eqdrift,equilibrium_sensitivity_pass=eqstable,
                        sensitivity_ranges_overlap=overlap,numerical_agreement_supported=supported,
                        evidence_class="finite_window_numerical_support" if supported else "time_checks_pass_comparison_unresolved" if selected and passed else
                        "confirmation_time_side_unresolved" if selected else "pilot_only_not_sufficiently_confirmed")
        check_fields(audit,reported[name,],expected,("primary",name))
        results.append(dict(pattern=name,**expected))
    return results


def audit_ranking(audit,names,tf,pairs,slopes):
    reported=keyed(audit.csv("pilot_candidate_ranking.csv"),("pattern",))
    expected=[]
    for name in names:
        pair=next((r for r in pairs if r["pattern"]==name and r["L_small"]==72 and r["L_large"]==108 and r["window"]=="4-9"),None)
        main=next((r for r in tf if r["pattern"]==name and r["L"]==108 and r["window"]=="4-9" and r["inverse_order"]==2),None)
        models=[r for r in tf if r["pattern"]==name and r["L"]==108 and r["window"]=="4-9" and r["formal_robustness_eligible"]]
        windows=[r for r in tf if r["pattern"]==name and r["L"]==108 and r["inverse_order"]==2 and r["complete_grid"]]
        local=[r["log_slope"] for r in slopes if r["pattern"]==name and r["domain"]=="time" and r["L"]==108 and
               r["pair_kind"]=="adjacent" and r["x_low"]>=4 and r["x_high"]<=9]
        c=main["c_fit"] if main else None
        md=max((ratio(r["c_fit"]-c,c) for r in models),default=None) if c else None
        wd=max((ratio(r["c_fit"]-c,c) for r in windows),default=None) if c else None
        size=pair.get("raw_size_difference_over_D_span") if pair else None
        score=max(size/.02,md/.05,wd/.05) if all(v is not None for v in (size,md,wd)) else None
        row=dict(pattern=name,pilot_L_small=72,pilot_L_large=108,pilot_window="4-9",pilot_common_complete=bool(pair and pair["complete_grid"]),
                 pilot_score=score,raw_size_difference_over_D_span=size,model_drift=md,window_drift=wd,c_time_fit=c,
                 raw_local_slope_min=min(local,default=None),raw_local_slope_max=max(local,default=None),
                 raw_local_slope_first=local[0] if local else None,raw_local_slope_last=local[-1] if local else None,
                 model_count=len(models),window_count=len(windows),scale_screen=False,equilibrium_used_for_ranking=False,
                 status="exploratory_candidate_not_confirmed" if pair and pair["complete_grid"] else "incomplete_pilot_unresolved")
        expected.append(row)
    expected.sort(key=lambda r:(not r["pilot_common_complete"],r["pilot_score"] is None,r["pilot_score"] if r["pilot_score"] is not None else math.inf,r["pattern"]))
    audit.exact(set(reported),{(n,) for n in names},("pilot ranking","patterns"))
    for rank,row in enumerate(expected,1):
        row["exploratory_rank"]=rank
        check_fields(audit,reported[row["pattern"],],row,("pilot ranking",row["pattern"]))


def counterfactual_equilibrium_test(audit,names,tf,ef,pairs,confirmation):
    # Use the production selector only for a separate mutation test, never as the
    # reference fit or reference gate. All input fits below were independently rebuilt.
    path=ROOT/"analyze_beta0_study.py"
    audit.read(path)
    spec=importlib.util.spec_from_file_location("_beta0_selector_for_mutation_audit",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    baseline=module.eligibility(names,tf,ef,pairs,confirmation)
    variants=[[],[dict(r,c_fit=(i+1)*10000. * (-1 if i%2 else 1)) for i,r in enumerate(ef)]]
    fields=list(choose_time(names[0],tf,pairs))+["confirmation_selected","confirmation_local_checks_pass","status"]
    for variant in variants:
        altered=module.eligibility(names,tf,variant,pairs,confirmation)
        for before,after in zip(baseline,altered):
            for field in fields:
                audit.exact(before[field],after[field],("equilibrium mutation",before["pattern"],field))
    return dict(variants=2,patterns=len(names),fields=fields,result="PASS",
                meaning="Deleting limit fits or replacing every equilibrium c by alternating large arbitrary values leaves all time-side fields unchanged")


def self_test():
    mp=load_mp()
    fit=ReferenceFits(mp)
    points=[]
    for i in range(8,21):
        x=mp.mpf(i)/2
        y=mp.mpf("0.1558")*mp.log(x)+mp.mpf("0.37")+mp.mpf("0.2")/x-mp.mpf("0.03")/x**2+mp.mpf("0.004")/x**3
        points.append(dict(theta=float(x),D_decimal=mp.nstr(y,80)))
    got=fit.fit(points,"theta",3)
    for field,want in zip(("c_fit","constant","inv1","inv2","inv3"),(.1558,.37,.2,-.03,.004)):
        assert abs(got[field]-want)<1e-14,(field,got[field],want)
    assert got["max_abs_residual"]<1e-65
    secant=(mp.mpf(points[-1]["D_decimal"])-mp.mpf(points[0]["D_decimal"]))/mp.log(mp.mpf("2.5"))
    assert abs(got["raw_endpoint_secant"]-float(secant))<1e-15
    print("PASS: synthetic 80-digit rectangular QR; no data receipt written")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:
        self_test()
        return
    audit=Audit()
    summary=audit.js(OUT/"study_summary.json")
    audit.exact(summary["status"],"PASS","analysis status")
    audit.exact(summary["stage"],"final","Refuse to certify in-progress analysis")
    assert summary["source_run_status"] and all(s=="complete" for s in summary["source_run_status"].values())
    for path,sha in summary["files"].items():
        audit.exact(digest(audit.read(path)),sha,("analysis input hash",path))
    for source in summary["raw_source_snapshots"]:
        audit.exact(digest(audit.read(source["original_path"])),source["sha256"],("live source vs final snapshot",source["original_path"]))
    data_receipt=audit.js(OUT/"final_data_validation.json")
    audit.exact(data_receipt["status"],"PASS","data validation status")
    audit.exact(data_receipt["stage"],"final","data validation stage")
    for path,sha in data_receipt["files"].items():
        audit.exact(digest(audit.read(path)),sha,("data validation hash",path))
    design=audit.js(OUT/"study_design.json")
    confirmation=audit.js(OUT/"confirmation_plan.json")
    names=design["patterns"]
    selected=confirmation.get("patterns",confirmation.get("selected_patterns",[]))
    audit.exact(len(names),16,"frozen candidate count")
    audit.exact(len(selected),10,"confirmation cohort count")
    assert len(set(selected))==10 and set(selected)<=set(names)
    audit.exact(set(confirmation["not_confirmed"]),set(names)-set(selected),"pilot-only cohort")
    for path,sha in confirmation["sources"].items():
        audit.exact(digest(audit.read(path)),sha,("frozen confirmation source",path))
    audit.exact(confirmation["primary"],dict(L=216,comparison_L=180,theta_min=4,theta_max=9,theta_step=.5,model="log_inv1_inv2"),"frozen primary anchor")
    audit.exact(confirmation["windows"],[list(w) for w in WINDOWS],"frozen windows")
    audit.exact(confirmation["model_orders"],[1,2,3],"frozen model orders")
    audit.exact(confirmation["order3_minimum_points"],11,"frozen order3 minimum")
    audit.exact(confirmation["thresholds"],dict(theta_min=4,theta_over_L_max=.05,size_coefficient_drift=.02,
                D_difference_over_fitted_log_span=.01,model_drift=.05,window_drift=.05,
                equilibrium_sensitivity=.05,postselection_relative_eq_time_gap=.05),"frozen thresholds")
    metadata=audit.js(OUT/"baseline_provenance.json")["patterns"]
    observations=[]
    seen=set()
    for row in audit.csv("study_observations.csv"):
        p=dict(row,L=int(row["L"]),theta=float(row["theta"]),D=float(row["D"]),D_decimal=row["D"])
        key=p["pattern"],p["L"],p["theta"]
        assert key not in seen,key
        seen.add(key)
        audit.number(row["beta"],0,("beta",key),atol=0,rtol=0)
        if p["pattern"] in names:
            assert truth(row["ground_reachable"]) and metadata[p["pattern"]]["ground_reachable"]
        observations.append(p)
    required_output=audit.csv("study_required_coverage.csv")
    required_keys=[]
    for filename in ("pilot_jobs.csv","pilot_reused.csv","confirmation_jobs.csv","confirmation_reused.csv"):
        for r in audit.csv(filename):
            k=r["pattern"],int(r["L"]),float(r["theta"])
            assert k in seen,("Missing frozen target",filename,k)
            required_keys.append((filename,*k))
    audit.exact(len(required_output),len(required_keys),"required coverage count")
    actual_required=[(r["plan_file"],r["pattern"],int(r["L"]),float(r["theta"])) for r in required_output]
    audit.exact(sorted(actual_required),sorted(required_keys),"required coverage target keys")
    assert all(truth(r["present"]) for r in required_output)
    audit.exact(summary["counts"]["missing_required_keys"],0,"final required targets complete")
    audit.exact(summary["confirmation_patterns"],selected,"summary confirmation cohort")
    conventions=summary["conventions"]
    audit.exact(conventions["primary"],dict(L=216,comparison_L=180,window=[4,9],model="log_inv1_inv2"),"primary specification")
    audit.exact(conventions["time_windows"],[list(w) for w in WINDOWS],"seven predeclared windows")
    audit.exact(conventions["threshold"],dict(size_drift=.02,D_difference_to_fitted_log_span=.01,model_drift=.05,window_drift=.05),"time-side thresholds")
    audit.exact(conventions["five_parameter_minimum_points"],11,"order3 minimum points")
    audit.exact(conventions["third_size_window"],[4,7],"third-size window")
    audit.exact(conventions["third_sizes"],[144,180,216],"third-size grid")
    mp=load_mp()
    fitter=ReferenceFits(mp)
    times,limits,tf,ef,pairs=reconstruct(audit,fitter,observations,names,metadata)
    slopes=audit_slopes(audit,observations,names,metadata,mp)
    audit_ranking(audit,names,tf,pairs,slopes)
    primary=audit_primary(audit,names,tf,ef,pairs,selected)
    mutation=counterfactual_equilibrium_test(audit,names,tf,ef,pairs,selected)
    availability=[]
    for name in names:
        sizes=sorted(L for n,L in times if n==name and times[n,L])
        complete=sorted(L for L in sizes if all(t in times[name,L] for t in required(4,9)))
        availability.append(dict(pattern=name,cohort="confirmation" if name in selected else "pilot_only",
                                 finite_sizes=sizes,max_finite_L=max(sizes,default=None),complete_primary_window_sizes=complete,
                                 max_complete_primary_window_L=max(complete,default=None),
                                 equilibrium_sizes=sorted(p["L"] for p in limits[name])))
    def common(cohort):
        sets=[set(r["complete_primary_window_sizes"]) for r in availability if cohort=="all" or r["cohort"]==cohort]
        shared=sorted(set.intersection(*sets)) if sets else []
        return dict(complete_window_L=shared,largest_common_L=max(shared,default=None))
    audit.exact(summary["counts"]["time_fits"],len(tf),"summary time fit count")
    audit.exact(summary["counts"]["equilibrium_fits"],len(ef),"summary limit fit count")
    audit.exact(summary["counts"]["raw_local_slopes"],len(slopes),"summary raw slope count")
    audit.read(__file__)
    audit.unchanged()
    receipt=dict(status="PASS",stage="final",generated_utc=datetime.now(timezone.utc).isoformat(),
                 scope="Independent fit arithmetic, actual grids, anchors, raw log secants, time-only eligibility and cohort coverage; not proof of asymptotic scaling",
                 method="80 decimal digit mpmath rectangular QR on CSV decimal observations; independent reconstruction without analysis fitting helpers",
                 decimal_digits=80,numeric_tolerance=dict(absolute=ATOL,relative=RTOL),
                 counts=dict(checks=audit.checks,unique_QR_fits=len(fitter.cache),time_fits=len(tf),equilibrium_fits=len(ef),
                             raw_slopes=len(slopes),paired_size_checks=len(pairs),patterns=len(names),confirmation_patterns=len(selected),pilot_only_patterns=len(names)-len(selected)),
                 max_absolute_errors=dict(audit.errors),worst_cases=audit.worst,
                 time_only_mutation_test=mutation,availability=availability,
                 shared_time_grid={cohort:common(cohort) for cohort in ("all","confirmation","pilot_only")},
                 primary=primary,files=audit.files,
                 interpretation="PASS certifies reproducibility and labeling, not that every fit passes the time-side criteria. Missing confirmation is unresolved, not a failed physical scaling law.")
    target=OUT/"fit_validation.json"
    temporary=target.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    temporary.replace(target)
    print(json.dumps(dict(status="PASS",**receipt["counts"],max_absolute_errors=receipt["max_absolute_errors"],shared_time_grid=receipt["shared_time_grid"]),indent=2))


if __name__=="__main__":
    main()
