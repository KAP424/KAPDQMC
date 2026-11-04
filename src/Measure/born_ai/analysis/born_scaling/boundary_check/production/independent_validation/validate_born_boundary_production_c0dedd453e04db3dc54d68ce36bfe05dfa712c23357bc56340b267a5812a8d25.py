"""Independent audit prepared before boundary-check production.

No sampling and no production-analyzer imports. Default execution requires
completed plan/manifest and final analysis. --self-test uses synthetic arrays.
Every translation stays inside its trajectory; all contrasts are paired before
estimating errors. The already independently audited statistics implementation
is loaded from its content-addressed snapshot. Fits use 80 decimal digits.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import tomllib

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "ai/analysis_beta_patterns/audit_dependencies"))
import mpmath as mp

DATA = ROOT / "data/born_scaling/boundary_check/production"
ANALYSIS = ROOT / "analysis/born_scaling/boundary_check/production"
OUT = ANALYSIS / "independent_validation"
SIZES = [6, 12, 18, 24, 30, 36]
REGIONS = ("SA", "SB", "SC", "D")
BLOCKS = (16, 32, 64, 128)
IDENTITIES = {"p2_group_minus_origin_SA", "p2_group_minus_origin_SC"}
TERMS = {"constant": ("intercept",), "log": ("intercept", "logL"),
         "linear": ("intercept", "L"), "invL": ("intercept", "invL"),
         "log_invL": ("intercept", "logL", "invL")}
HASHES, MAX_ERRORS = {}, {}
CHECKS = 0


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path):
    return Path(os.path.relpath(Path(path).resolve(), ROOT)).as_posix()


def bind(path, expected=None):
    actual = sha(path)
    assert expected is None or actual == expected, ("SHA mismatch", path)
    HASHES[relative(path)] = actual
    return actual


def mapping(value):
    for name, digest in value.items():
        bind(ROOT/name, digest)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_toml(path):
    return tomllib.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    bind(path)
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def boolean(value):
    assert value in (True, False, "True", "False", "true", "false"), value
    return value in (True, "True", "true")


def close(name, actual, expected, atol=3e-12, rtol=3e-9):
    global CHECKS
    CHECKS += 1
    if expected is None:
        assert actual in (None, ""), (name, actual)
        return
    assert actual not in (None, ""), (name, actual, expected)
    actual, expected = float(actual), float(expected)
    difference = abs(actual-expected)
    MAX_ERRORS[name] = max(MAX_ERRORS.get(name, 0.), difference)
    assert math.isfinite(actual) and difference <= atol+rtol*abs(expected), (name, actual, expected)


def load_independent_statistics():
    receipt_path = ROOT/"analysis/born_scaling_production_validation.json"
    receipt = read_json(receipt_path)
    assert receipt["status"] == "PASS"
    path = ROOT/receipt["validator_snapshot"]
    bind(path, receipt["files"][receipt["validator_snapshot"]])
    bind(receipt_path)
    spec = importlib.util.spec_from_file_location("frozen_independent_born_statistics", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def chain_records(job):
    """Read thirteen rows per recorded state; never flatten shifts into draws."""
    path = ROOT/job["sample_path"]
    raw = rows(path)
    L, step, burn = job["L"], job["measure_every"], job["warmup_sweeps"]
    assert len(raw) == (job["sweeps"]//step)*13
    grouped = {}
    for row in raw:
        for field in ("L", "chain", "seed"):
            assert int(row[field]) == job[field]
        assert float(row["beta"]) == 3. and float(row["mu"]) == 1e-8
        assert row["stage"] == "production" and row["start"] == job["start"]
        record, sweep = int(row["record_index"]), int(row["sweep"])
        assert sweep == record*step
        assert int(row["measure_every"]) == step
        den, tx, ty = int(row["p_den"]), int(row["tx"]), int(row["ty"])
        assert int(row["p_num"]) == 1 and den in (3, 2)
        assert tx in range(0, L, L//den) and ty in range(0, L, L//den)
        assert int(row["translation_index"]) == 1+tx//(L//den)+den*(ty//(L//den))
        entropy = np.array([float(row[name]) for name in REGIONS])
        assert np.isfinite(entropy).all()
        close("per_row_D_identity", entropy[3], (1-1/den)*entropy[0]+entropy[2]/den-entropy[1])
        metadata = (sweep, int(row["accepted"]), int(row["proposed"]), float(row["log_weight"]))
        assert metadata[2] == 2*L*L*step and 0 <= metadata[1] <= metadata[2]
        entry = grouped.setdefault(record, {"metadata":metadata, "shifts":{}})
        assert entry["metadata"] == metadata
        key = (den, tx, ty)
        assert key not in entry["shifts"]
        entry["shifts"][key] = entropy
    assert sorted(grouped) == list(range(1, job["sweeps"]//step+1))
    expected = {(m, tx, ty) for m in (3,2) for ty in range(0,L,L//m) for tx in range(0,L,L//m)}
    retained = []
    for record in sorted(grouped):
        entry = grouped[record]
        assert set(entry["shifts"]) == expected
        if entry["metadata"][0] > burn:
            retained.append(entry)
    assert len(retained) == job["retained_records"]
    return retained


def paired_observables(chains, L):
    """Independent scalar assembly, with four chain axes retained throughout."""
    result = {}
    for m in (3,2):
        shifts = [(m,tx,ty) for ty in range(0,L,L//m) for tx in range(0,L,L//m)]
        for ri, region in enumerate(REGIONS):
            origin = np.array([[entry["shifts"][(m,0,0)][ri] for entry in chain] for chain in chains])
            group = np.array([[np.mean([entry["shifts"][shift][ri] for shift in shifts])
                               for entry in chain] for chain in chains])
            result[f"p{m}_origin_{region}"] = origin
            result[f"p{m}_group_{region}"] = group
            result[f"p{m}_group_minus_origin_{region}"] = group-origin
        for ti, shift in enumerate(shifts):
            result[f"p{m}_shift{ti}_D"] = np.array([[entry["shifts"][shift][3] for entry in chain] for chain in chains])
    for scope in ("origin", "group"):
        for region in REGIONS:
            result[f"half_minus_third_{scope}_{region}"] = result[f"p2_{scope}_{region}"]-result[f"p3_{scope}_{region}"]
    return result


def mp_fit(sizes, values, errors, terms):
    with mp.workdps(80):
        grid = [mp.mpf(str(v)) for v in sizes]
        sigma = [mp.mpf(str(v)) for v in errors]
        assert all(v > 0 for v in sigma)
        X = mp.matrix([[{"intercept":mp.mpf(1), "L":L, "logL":mp.log(L), "invL":1/L}[t]
                        for t in terms] for L in grid])
        y = mp.matrix([mp.mpf(str(v)) for v in values])
        A = mp.matrix([[X[i,j]/sigma[i] for j in range(X.cols)] for i in range(X.rows)])
        c, _ = mp.qr_solve(A, mp.matrix([y[i]/sigma[i] for i in range(len(y))]))
        cov = (A.T*A)**-1
        residual = y-X*c
        chi = sum((residual[i]/sigma[i])**2 for i in range(len(y)))
        return dict(coefficient={t:float(c[i]) for i,t in enumerate(terms)},
                    coefficient_MCSE={t:float(mp.sqrt(cov[i,i])) for i,t in enumerate(terms)},
                    coefficient_covariance=[[float(v) for v in row] for row in cov.tolist()],
                    residuals=[float(v) for v in residual],
                    weighted_residual_sum_squares=float(chi), reduced_chi_square=float(chi/(len(y)-len(terms))))


def validate_covariance(all_values, table):
    indexed = {(int(r["L"]), int(r["block_length"]), r["first"], r["second"]):r for r in table}
    assert len(indexed) == len(table)
    checked = []
    for L, values in all_values.items():
        names = [n for n in values if "_shift" not in n]
        draws = next(iter(values.values())).shape[1]
        for length in BLOCKS:
            blocks = draws//length
            if blocks < 8 or (draws-blocks*length)/draws > .05:
                continue
            # Build blocks with independent explicit slicing instead of the
            # analyzer's joint-array reshape. Covariances retain all pairing.
            B = np.array([[float(values[name][chain,b*length:(b+1)*length].mean()) for name in names]
                          for chain in range(4) for b in range(blocks)])
            center = B-B.mean(axis=0)
            covariance = center.T@center/(len(B)*(len(B)-1))
            for i, first in enumerate(names):
                for j, second in enumerate(names):
                    row = indexed[(L,length,first,second)]
                    assert boolean(row["eligible"])
                    close("block_covariance", row["covariance_mean"], covariance[i,j], atol=2e-13, rtol=1e-8)
            a, b, diff = (names.index(n) for n in ("p2_group_D","p3_group_D","half_minus_third_group_D"))
            close("paired_variance_identity", covariance[diff,diff],
                  covariance[a,a]+covariance[b,b]-2*covariance[a,b], atol=2e-13)
            checked.append(dict(L=L,block_length=length,blocks_per_chain=blocks,
                                paired_shape_D_variance=float(covariance[diff,diff])))
        assert any(r["L"] == L for r in checked), (L,"no eligible covariance block")
    return checked


def old_comparison(means, new_seeds):
    summary_path = ROOT/"analysis/born_scaling/beta3_large/summary.json"
    old_summary = read_json(summary_path)
    assert old_summary["status"] == "PASS"
    bind(summary_path)
    path = ROOT/"analysis/born_scaling/beta3_large/means.csv"
    bind(path, old_summary["files"][relative(path)])
    old = {int(r["L"]):r for r in rows(path)}
    old_seeds = set()
    for rel in ("data/born_scaling/production/production_plan.toml",
                "data/born_scaling/production_beta3_large/production_beta3_large_plan.toml"):
        p = ROOT/rel
        bind(p)
        old_seeds.update(j["seed"] for j in read_toml(p)["jobs"])
    assert not new_seeds.intersection(old_seeds), "New and old seeds overlap"
    comparison = []
    for L in SIZES:
        for scope in ("group", "origin"):
            for observable in ("D", "SA", "SB", "SC"):
                r = means[(L,f"p3_{scope}_{observable}")]
                se = float(r["MCSE"])
                combined = math.hypot(se,float(old[L][f"{observable}_MCSE"]))
                difference = float(r["mean"])-float(old[L][f"{observable}_mean"])
                comparison.append(dict(L=L,scope=scope,observable=observable,new_mean=float(r["mean"]),
                    old_mean=float(old[L][f"{observable}_mean"]),difference=difference,
                    combined_MCSE=combined,z=difference/combined,
                    caution="Different-seed production; standardized discrepancy is descriptive, with many correlated comparisons"))
    return comparison


def self_test():
    helper = load_independent_statistics()
    L, n = 6, 256
    chains = []
    for chain in range(4):
        entries = []
        for i in range(n):
            t = math.sin((.217+.013*chain)*i)+.1*math.cos(.093*i)
            shifts = {}
            for m in (3,2):
                for ty in range(0,L,L//m):
                    for tx in range(0,L,L//m):
                        a = 2+t+(.01*(tx+ty) if m==3 else 0.)
                        c = 2.2+.7*t+(.02*(tx-ty) if m==3 else 0.)
                        b = 1.8+.6*t+.015*(tx-ty)
                        shifts[(m,tx,ty)] = np.array([a,b,c,(1-1/m)*a+c/m-b])
            entries.append(dict(shifts=shifts))
        chains.append(entries)
    values = paired_observables(chains,L)
    assert all(x.shape==(4,n) for x in values.values())
    assert max(np.max(abs(values[name])) for name in IDENTITIES) < 1e-14
    d,_ = helper.diagnostics(values["p3_group_D"])
    assert d["MCSE_conservative"] > 0
    grid = SIZES
    truth = [.7+.155*math.log(L)+1.3/L for L in grid]
    fit = mp_fit(grid,truth,[.002]*len(grid),TERMS["log_invL"])
    assert max(abs(fit["coefficient"][name]-val) for name,val in zip(TERMS["log_invL"],[.7,.155,1.3])) < 1e-12
    contrast = [.04+.023*math.log(L)-.2/L for L in grid]
    fit = mp_fit(grid,contrast,[.001]*len(grid),TERMS["log_invL"])
    assert abs(fit["coefficient"]["logL"]-.023) < 1e-12
    print("PASS: synthetic 13-row pairing, pure-complement identity and 80-digit log+1/L/paired-contrast fits; no new production files read")


def main():
    target = OUT/"validation.json"
    if target.exists():
        raise FileExistsError("Preserve existing independent validation")
    plan_path, manifest_path, summary_path = DATA/"plan.toml", DATA/"manifest.json", ANALYSIS/"summary.json"
    assert all(p.exists() for p in (plan_path,manifest_path,summary_path)), "Wait for complete production and analysis"
    plan, manifest, summary = read_toml(plan_path), read_json(manifest_path), read_json(summary_path)
    assert plan["stage"] == manifest["stage"] == summary["stage"] == "production"
    assert manifest["status"] == "complete" and summary["status"] == "PASS"
    bind(plan_path,manifest["plan_sha256"]); bind(manifest_path); bind(summary_path)
    for obj in (plan,manifest,summary):
        for field in ("files","outputs","source_sha256","evidence_sha256"):
            mapping(obj.get(field,{}))
    for name in ("allocation", "pilot_summary"):
        if name+"_path" in plan:
            bind(ROOT/plan[name+"_path"], plan[name+"_sha256"])
    helper = load_independent_statistics()
    grouped, seeds, kept = {}, set(), 0
    for job in plan["jobs"]:
        assert job["seed"] not in seeds
        seeds.add(job["seed"])
        assert job["stage"] == "production" and job["L"] in SIZES
        receipt_path = ROOT/job["receipt_path"]
        receipt = read_toml(receipt_path)
        assert receipt["status"] == "complete" and receipt["job"] == job
        assert receipt["plan_sha256"] == sha(plan_path)
        for key in ("maximum_inverse_drift", "maximum_logweight_drift"):
            assert 0 <= receipt[key] < 1e-5
        bind(receipt_path)
        mapping(receipt["files"])
        entries = chain_records(job)
        kept += len(entries)
        grouped.setdefault(job["L"],[]).append((job["chain"],entries))
    assert sorted(grouped) == SIZES and kept == manifest["retained_records"] == summary["retained_records"]
    all_values = {}
    for L, chains in grouped.items():
        chains.sort(key=lambda item:item[0])
        assert [i for i,_ in chains] == [1,2,3,4]
        assert len({len(records) for _,records in chains}) == 1
        all_values[L] = paired_observables([records for _,records in chains],L)
    mean_rows = [r for r in rows(ANALYSIS/"means.csv") if float(r["beta"]) == 3.]
    means = {(int(r["L"]),r["observable"]):r for r in mean_rows}
    assert len(means) == len(mean_rows)
    diag_rows = rows(ANALYSIS/"diagnostics.csv")
    diagnostics = {(int(r["L"]),r["observable"]):r for r in diag_rows}
    identity_checks = []
    for L, values in all_values.items():
        for name, x in values.items():
            r = means[(L,name)]
            close("mean",r["mean"],x.mean())
            close("record_count",r["draws_per_chain"],x.shape[1])
            close("chain_count",r["n_chains"],4)
            if name in IDENTITIES:
                assert np.max(abs(x)) < 1e-9
                identity_checks.append(dict(L=L,observable=name,maximum_absolute_difference=float(np.max(abs(x))),
                                            interpretation="Exact pure-state complementary half-strip identity, up to numerical error"))
                continue
            assert np.ptp(x)>1e-12, (L,name,"Unexpected constant observable requires explicit review")
            expected,_ = helper.diagnostics(x)
            for field in ("MCSE_conservative","rhat_max","ESS_bulk","ESS_tail","ESS_mean","MCSE_ess"):
                close("diagnostic_"+field,diagnostics[(L,name)][field],expected[field])
            close("mean_table_MCSE",r["MCSE"],expected["MCSE_conservative"])
            assert boolean(r["screen_pass"]) == expected["initial_screen_pass"]
    checked_cov = validate_covariance(all_values,rows(ANALYSIS/"covariance.csv"))
    fit_path = ANALYSIS/"fits.json"
    fits = read_json(fit_path); bind(fit_path)
    for name in ("p3_group_D","p2_group_D","half_minus_third_group_D"):
        required = [f for f in fits if float(f["beta"])==3. and f["observable"]==name
                    and f["model"]=="log_invL" and f["requested_L_min"]==6]
        assert len(required)==1, (name,"Missing or duplicate preregistered full-window fit")
        if all(boolean(means[(L,name)]["screen_pass"]) for L in SIZES):
            assert required[0]["status"]=="finite_window_diagnostic_only", (name,"Eligible full-window fit was omitted")
        else:
            assert required[0]["status"] in ("sampling_screen_not_passed","sampling_uncertainty_unresolved")
    for reported in summary.get("paired_corner_coefficient_diagnostics",[]):
        source = [f for f in fits if f["beta"]==reported["beta"] and f["observable"]==reported["observable"]
                  and f["model"]==reported["model"] and f["requested_L_min"]==reported["requested_L_min"]]
        assert len(source)==1 and source[0]["status"]==reported["status"]
        if reported["status"]=="finite_window_diagnostic_only":
            close("summary_paired_delta_c",reported["coefficient"]["logL"],source[0]["coefficient"]["logL"])
    fit_checks, paired_corner = [], []
    for fit in fits:
        if float(fit["beta"]) != 3. or fit["status"] != "finite_window_diagnostic_only":
            continue
        name, terms = fit["observable"], TERMS[fit["model"]]
        sizes = [int(L) for L in fit["sizes"]]
        assert sizes == [L for L in SIZES if L>=fit["requested_L_min"]]
        assert len(sizes)-len(terms)>=3
        assert all(boolean(means[(L,name)]["screen_pass"]) for L in sizes)
        calc = mp_fit(sizes,[all_values[L][name].mean() for L in sizes],
                      [float(means[(L,name)]["MCSE"]) for L in sizes],terms)
        for field in ("coefficient","coefficient_MCSE"):
            for term in terms:
                close("fit_"+field,fit[field][term],calc[field][term],atol=2e-10,rtol=2e-8)
        for field in ("weighted_residual_sum_squares","reduced_chi_square"):
            close("fit_"+field,fit[field],calc[field],atol=2e-8,rtol=2e-8)
        fit_checks.append(dict(observable=name,model=fit["model"],L_min=min(sizes),L_max=max(sizes),**calc))
        if name in ("p3_group_D","p2_group_D","half_minus_third_group_D") and fit["model"]=="log_invL" and min(sizes)==6:
            paired_corner.append(dict(observable=name,**calc))
    # Screen failures are reported as missing usable fits, never repaired by
    # fitting excluded points or changing windows during this independent audit.
    result = dict(status="PASS",created_utc=datetime.now(timezone.utc).isoformat(),
        retained_records=kept,raw_retained_entropy_rows=13*kept,n_chains=len(plan["jobs"]),
        numerical_comparisons=CHECKS,maximum_absolute_errors=MAX_ERRORS,
        complement_identity_checks=identity_checks,eligible_block_covariance_checks=checked_cov,
        high_precision_fit_checks=fit_checks,full_window_log_invL_corner_checks=paired_corner,
        old_new_production_comparison=old_comparison(means,seeds),
        limitations=["Audit verifies arithmetic, pairing and file provenance; it does not certify Monte Carlo tail coverage or an asymptotic scaling law.",
                     "Old/new standardized differences are descriptive and multiple correlated comparisons are shown.",
                     "Pure-complement strip contrasts are identities, not failed stochastic sampling screens."],files=HASHES)
    OUT.mkdir(parents=True,exist_ok=True)
    snapshot = OUT/(Path(__file__).stem+"_"+sha(__file__)+".py")
    snapshot.write_bytes(Path(__file__).read_bytes())
    result["validator_snapshot"] = relative(snapshot)
    result["validator_sha256"] = bind(snapshot)
    with target.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ("status","retained_records","n_chains","numerical_comparisons","maximum_absolute_errors")},indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test",action="store_true")
    args = parser.parse_args()
    self_test() if args.self_test else main()
