"""Independent audit of additional pure-0/2 patterns, including blocked states.

Default mode requires complete data and a current final analysis. Development
mode (--allow-incomplete) never publishes final PASS. Only generic snapshot,
hashing and mpmath-loading utilities are reused from the previous validator;
no simulation or production fitting routine is imported or called.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import re
import tomllib

from validate_irregular_followup import Snapshot, sha, key, json_key, truth, load_mpmath

ROOT = Path(__file__).resolve().parent
DATA = ROOT/"data"
OUT = ROOT/"analysis_second_class_extra"
MODELS = {"log_inv1": (1,), "log_inv1_inv2": (1, 2), "log_inv1_inv2_inv3": (1, 2, 3)}


def physical_proof(snap, metadata):
    proof_path = DATA/"second_class_extra_validation.toml"
    proof = tomllib.loads(snap.read(proof_path).decode())
    assert proof["status"] == "PASS" and proof["states"] == 48
    assert sha(snap.read(DATA/"second_class_extra_validation.csv")) == proof["csv_sha256"]
    for filename, digest in proof["code_sha256"].items():
        assert sha(snap.read(ROOT/filename)) == digest, (filename, "changed since small-system validation")
    _, rows, _ = snap.table(DATA/"second_class_extra_validation.csv")
    assert len(rows) == 48
    assert {(r["pattern"], float(r["theta"]), float(r["beta"])) for r in rows} == set(
        itertools.product(metadata, (0., 4., 7., math.inf), (0., 1., 2.)))
    summaries = {r["pattern"]: r for r in proof["patterns"]}
    assert set(summaries) == set(metadata)
    maxima = {"max_covariance_error": "covariance_error", "max_covariance_element_error": "covariance_element_error",
              "max_entropy_error": "entropy_error", "max_D_error": "D_error", "max_finite_step_difference": "finite_step_difference"}
    for out, column in maxima.items():
        value = max(float(r[column]) for r in rows)
        assert value == proof[out] and math.isfinite(value) and value < 1e-8, (out, value)
    infinite_maxima = {out: max(float(r[column]) for r in rows if math.isinf(float(r["theta"])))
                       for out, column in maxima.items() if out != "max_finite_step_difference"}
    assert all(value < 1e-10 for value in infinite_maxima.values())
    for name, meta in metadata.items():
        summary = summaries[name]
        reachable = truth(meta["ground_reachable"])
        assert summary["occupied_bits"] == meta["occupied_bits"] and summary["ground_reachable"] == reachable
        assert summary["limit_state"] == meta["limit_state"]
        assert summary["L"] == (12 if meta["m"] == 4 else 6)
        assert summary["independent_limit_residual"] < 1e-9
        excess = summary["independent_limit_excess"]
        assert abs(excess) < 1e-8 if reachable else excess > 1e-6
        assert summary["max_discarded_sv"] < 1e-10 and summary["min_retained_sv"] > 1e-8
    for row in rows:
        meta = metadata[row["pattern"]]
        assert truth(row["ground_reachable"]) == truth(meta["ground_reachable"])
        assert row["limit_state"] == meta["limit_state"]
        assert row["reference_method"] == ("realspace_energy_block_nullspace" if math.isinf(float(row["theta"])) else "realspace_exp_qr")
        assert float(row["entropy_error"]) == max(abs(float(row[f])-float(row["reference_"+f])) for f in ("S_A", "S_C", "S_B"))
        assert float(row["D_error"]) == abs(float(row["D"])-float(row["reference_D"]))
    return dict(status="PASS", states=48, receipt_sha256=sha(snap.read(proof_path)),
        csv_sha256=proof["csv_sha256"], code_sha256=proof["code_sha256"],
        maxima={field: proof[field] for field in maxima}, infinite_time_maxima=infinite_maxima,
        patterns=proof["patterns"], scope=proof["scope"])


def audit_data(snap, plan):
    metadata = plan["pattern_metadata"]
    assert len(metadata) == 4 and set(metadata) == set(plan["patterns"])
    declarations = {name: (int(m), bits, reachable == "true") for name, m, bits, reachable in re.findall(
        r'\("([^"]+)",\s*(\d+),\s*"([01]+)",\s*(true|false)\)',
        snap.read(ROOT/"second_class_extra_patterns.jl").decode())}
    assert set(declarations) == set(metadata)
    for name, meta in metadata.items():
        m, bits, reachable = declarations[name]
        assert (meta["m"], meta["occupied_bits"], truth(meta["ground_reachable"])) == (m, bits, reachable)
        assert len(bits) == 2*m*m and bits.count("1") == m*m
        assert all(bits[i] == bits[i+1] for i in range(0, len(bits), 2))
        assert meta["occupancy_class"] == 2
        assert meta["limit_state"] == ("ground_state" if reachable else "accessible_excited_state")
        assert meta["projection_method"] == "strict_spectral_support"
    assert sum(truth(m["ground_reachable"]) for m in metadata.values()) == 2
    for path, digest in plan.get("frozen_code_sha256", {}).items():
        assert sha(snap.read(ROOT/path)) == digest, (path, "frozen source changed")
    bound_validation_sources = {}
    for path in plan.get("validation_sources", []):
        proof_raw = snap.read(ROOT/path)
        proof = json.loads(proof_raw)
        assert proof["status"] == "PASS"
        for source, digest in proof["files"].items():
            assert sha(snap.read(ROOT/source)) == digest, (source, "support-validation source changed")
        bound_validation_sources[path] = sha(proof_raw)
    physical = physical_proof(snap, metadata)
    required = set()
    for kind in ("required_limit_grids", "required_finite_grids"):
        for grid in plan[kind]:
            required.update(itertools.product(grid.get("patterns", plan["patterns"]), grid["L"],
                map(float, grid.get("theta", ["Inf"])), grid.get("beta", plan["betas"])))
    assert len(required) == plan["expected_observation_count"]
    _, support_rows, _ = snap.table(DATA/"second_class_extra_reachability.csv")
    support = {(r["pattern"], int(r["L"])): r for r in support_rows}
    assert len(support) == len(support_rows)
    assert {(k[0], k[1]) for k in required} <= set(support)
    for (name, size), row in support.items():
        meta = metadata[name]
        reachable = truth(meta["ground_reachable"])
        assert row["occupied_bits"] == meta["occupied_bits"] and int(row["m"]) == meta["m"]
        assert int(row["blocks"]) == (size//meta["m"])**2
        assert truth(row["expected_ground"]) == truth(row["ground_reachable"]) == reachable
        deficit = int(row["negative_rank_deficiency"])+int(row["nonpositive_rank_deficiency"])
        assert deficit == 0 if reachable else deficit > 0
        assert int(row["blocked_momenta"]) == 0 if reachable else int(row["blocked_momenta"]) > 0
        assert abs(float(row["excess_energy"])) < 1e-8 if reachable else float(row["excess_energy"]) > 1e-8
    selected, input_receipts, runs, maxima, pending, duplicates = {}, {}, [], defaultdict(float), [], []

    def bounded(field, value, limit, identity):
        value = abs(float(value))
        assert math.isfinite(value) and value < limit, (identity, field, value, limit)
        maxima[field] = max(maxima[field], value)

    for relative in plan["inputs"]:
        path = ROOT/relative
        if not path.exists():
            assert snap.incomplete, (relative, "missing source")
            pending.append(relative)
            continue
        content, rows, lines = snap.table(path)
        ledger_path = path.with_name(path.stem+"_provenance.tsv")
        _, ledger, _ = snap.table(ledger_path, "\t")
        assert snap.incomplete or len(rows) == len(ledger)
        paired = min(len(rows), len(ledger))
        if len(rows) != len(ledger):
            pending.append(relative+": unpaired CSV/ledger tail")
        rows = rows[:paired]
        assert len({key(r) for r in rows}) == len(rows), "Duplicate source observation"
        run_groups = defaultdict(list)
        for number, (row, entry) in enumerate(zip(rows, ledger[:paired]), 2):
            k = key(row)
            assert k in required, (k, "unplanned observation")
            name, size, theta, beta = k
            meta = metadata[name]
            reachable = truth(meta["ground_reachable"])
            assert int(row["m"]) == meta["m"] and row["family"] == "general" and row["occupied_bits"] == meta["occupied_bits"]
            assert size % (3*meta["m"]) == 0 and truth(row["premeasurement_ground"]) == reachable
            assert float(row["rank_tol"]) == 1e-10
            assert row["source_kind"] == "computed" and row["entropy_method"] == "real_cholesky"
            assert int(entry["output_row"]) == number and entry["csv_row"] == lines[number-1]
            assert entry["row_sha256"] == sha(lines[number-1].encode())
            assert all(entry[f] == row[f] for f in ("run_id", "source_kind", "source_path", "source_row", "source_sha256"))
            sa, sc, sb, d = (float(row[f]) for f in ("S_A", "S_C", "S_B", "D"))
            assert all(math.isfinite(x) and x >= -1e-8 for x in (sa, sc, sb))
            bounded("D_identity", d-(2*sa+sc)/3+sb, 1e-9, k)
            bounded("delta_identity", float(row["delta"])-sa+sb, 1e-9, k)
            for field, limit in (("projector_error", 1e-10), ("reconstruction_error", 1e-9),
                    ("premeasurement_residual", 1e-9), ("rectangle_imag_error", 2e-10), ("rectangle_real_error_bound", 1e-9)):
                bounded(field, row[field], limit, k)
            excess = float(row["accessible_excess"])
            assert math.isfinite(excess)
            assert abs(excess) < 1e-8 if reachable else excess > 1e-8
            bounded("independent_support_energy_difference", excess-float(support[name, size]["excess_energy"]), 1e-8, k)
            bounded("support_limit_energy_identity", float(row["premeasurement_excess"])-excess, 1e-8, k)
            if math.isinf(theta) and beta == 0:
                bounded("unmeasured_limit_energy_identity", float(row["excess_energy"])-excess, 1e-8, k)
                bounded("unmeasured_limit_residual", row["residual"], 1e-9, k)
            for field in ("x0", "y0", "cut_x0", "cut_y0"):
                if row.get(field):
                    assert int(row[field]) == 1
            if k in selected:
                other = selected[k][0]
                error = max(abs(float(row[f])-float(other[f])) for f in ("S_A", "S_C", "S_B", "D"))
                assert error <= 1e-9
                duplicates.append(dict(key=json_key(k), maximum_error=error))
            selected[k] = row, str(path.resolve()), number
            run_groups[row["run_id"]].append((number, row))
        for rid, run_rows in run_groups.items():
            manifest_path = DATA/f"second_class_extra_run_{rid}.toml"
            manifest_raw = snap.read(manifest_path)
            manifest = tomllib.loads(manifest_raw.decode())
            assert manifest["run_id"] == rid and Path(manifest["output"]).resolve() == path.resolve()
            assert "Exact spectral-support finite theta or accessible Inf" in manifest["protocol"]
            assert "blocked Inf is an excited state" in manifest["protocol"]
            if manifest["status"] != "complete":
                assert snap.incomplete and manifest["status"] in ("running", "failed", "interrupted")
                pending.append(f"{rid}: {manifest['status']}")
            hashes = manifest["code_sha256"]
            for filename, digest in hashes.items():
                assert sha(snap.read(DATA/"provenance/code"/digest/filename)) == digest
                assert sha(snap.read(ROOT/filename)) == digest, (filename, "run source differs from final source")
                if filename in physical["code_sha256"]:
                    assert digest == physical["code_sha256"][filename]
            assert hashes["analysis_second_class_extra/plan.json"] == sha(snap.read(OUT/"plan.json"))
            requested = set(itertools.product(manifest["patterns"], manifest["L"], map(float, manifest["theta"]), manifest["beta"]))
            for _, row in run_rows:
                assert key(row) in requested and row["code_sha256"] == hashes["second_class_extra_experiments.jl"]
            if manifest["status"] == "complete":
                assert len(run_rows) == manifest["new_rows"]
                final_row = max(number for number, _ in run_rows)
                assert sha(b"".join(content.splitlines(keepends=True)[:final_row])) == manifest["output_sha256"]
                checks = {(g["pattern"], int(g["L"])): g for g in manifest["ground_checks"]}
                assert {(r["pattern"], int(r["L"])) for _, r in run_rows} <= set(checks)
                for (name, _), check in checks.items():
                    reachable = truth(metadata[name]["ground_reachable"])
                    assert check["actual_ground"] == check["expected_ground"] == reachable
                    assert check["limit_state"] == metadata[name]["limit_state"]
                    assert abs(check["excess"]) < 1e-8 if reachable else check["excess"] > 1e-8
                    bounded("manifest_limit_residual", check["residual"], 1e-9, rid)
            runs.append(dict(path=str(manifest_path.relative_to(ROOT)), sha256=sha(manifest_raw), status=manifest["status"], rows=len(run_rows)))
        input_receipts[relative] = dict(sha256=sha(snap.read(path)), ledger_sha256=sha(snap.read(ledger_path)), rows=len(rows))
    missing = required-set(selected)
    assert snap.incomplete or not missing, ("Missing planned data", len(missing))
    return selected, dict(required_rows=len(required), available_required_rows=len(required-missing),
        missing_required_rows=len(missing), missing_keys=[json_key(k) for k in sorted(missing)],
        input_receipts=input_receipts, audited_runs=runs, numerical_maxima=dict(maxima),
        consistent_duplicates=duplicates, pending_sources_or_runs=pending, physical_validation=physical,
        bound_support_validation_sources=bound_validation_sources, independent_support_rows=len(support_rows))


def audit_analysis(snap, plan, selected):
    if not (OUT/"summary.json").exists():
        assert snap.incomplete
        return dict(status="SKIPPED_NO_DEVELOPMENT_ANALYSIS"), {}
    summary_raw = snap.read(OUT/"summary.json")
    summary = json.loads(summary_raw)
    stale = []
    if summary["script_sha256"] != sha(snap.read(ROOT/"analyze_second_class_extra.py")):
        stale.append("Analysis source changed")
    if summary["plan_sha256"] != sha(snap.read(OUT/"plan.json")):
        stale.append("Plan changed")
    for source in summary["sources"]:
        path = Path(source["path"])
        if source.get("missing"):
            if path.exists(): stale.append(str(path)+": was missing")
        elif not path.exists() or sha(snap.read(path)) != source["sha256"]:
            stale.append(str(path)+": changed")
    if stale and snap.incomplete:
        return dict(status="SKIPPED_STALE_DEVELOPMENT_ANALYSIS", stale=stale, summary_sha256=sha(summary_raw)), {}
    assert not stale, stale
    if not snap.incomplete:
        assert summary["stage"] == "final" and summary["primary_complete"] and summary["missing_required_keys"] == 0
        assert not summary["source_changes_during_run"]
    receipt = json.loads(snap.read(OUT/"receipt.json"))
    if not snap.incomplete:
        assert receipt["status"] == "COMPLETE" and receipt["stage"] == "final"
    for filename, digest in receipt["files"].items():
        assert sha(snap.read(ROOT/filename)) == digest, (filename, "Analysis dependency/output changed")
    table_names = ("observations", "limit_fits", "time_individual_fits", "time_pair_fits", "primary_parameters", "local_log_slopes")
    tables = {name: snap.table(OUT/f"{name}.csv")[1] for name in table_names}
    points = {key(r): r for r in tables["observations"]}
    assert len(points) == len(tables["observations"]) and set(points) == set(selected)
    metadata = plan["pattern_metadata"]
    for k, row in points.items():
        source, path, number = selected[k]
        meta = metadata[k[0]]
        assert truth(row["ground_reachable"]) == truth(meta["ground_reachable"])
        assert row["limit_state"] == meta["limit_state"] and row["projection_method"] == "strict_spectral_support"
        assert row["state_role"] == (meta["limit_state"]+"_limit" if math.isinf(k[2]) else "finite_imaginary_time")
        assert row["occupied_bits"] == meta["occupied_bits"] and int(row["x0"]) == int(row["y0"]) == 1
        assert (ROOT/row["source_file"]).resolve() == Path(path) and int(row["source_row"]) == number
        assert row["source_sha256"] == sha(snap.read(path))
        assert all(float(row[f]) == float(source[f]) for f in ("S_A", "S_C", "S_B", "D"))
    for name in table_names[1:]:
        for row in tables[name]:
            meta = metadata[row["pattern"]]
            assert truth(row["ground_reachable"]) == truth(meta["ground_reachable"])
            assert row["limit_state"] == meta["limit_state"]
            if not truth(meta["ground_reachable"]):
                assert row["parameter_meaning"] == "accessible_excited_state_fit_not_ground_state_corner_coefficient"
                if "ground_state_local_checks_pass" in row:
                    assert not truth(row["ground_state_local_checks_pass"])

    mp = load_mpmath()
    mp.mp.dps = 80
    maxima, fit_counts, cache = defaultdict(float), Counter(), {}

    def close(label, expected, actual, absolute=1e-8, relative=1e-8):
        expected, actual = float(expected), float(actual)
        difference = abs(expected-actual)
        assert math.isfinite(actual) and difference <= absolute+relative*abs(expected), (label, expected, actual, difference)
        maxima[label] = max(maxima[label], difference)

    def check_fit(row, domain, size=None, small=False):
        name, beta, model = row["pattern"], float(row["beta"]), row["model"]
        grid = [float(x) for x in row["grid"].split(",")]
        assert len(grid) >= len(MODELS[model])+4 and grid == sorted(set(grid))
        if domain == "limit":
            low, high = float(row["requested_Lmin"]), float(row["requested_Lmax"])
            expected_grid = sorted(k[1] for k in points if k[0] == name and k[3] == beta and math.isinf(k[2]) and low <= k[1] <= high)
            assert truth(row["extent_complete"]) == (min(grid) == low and max(grid) == high)
        else:
            low, high = float(row["theta_min"]), float(row["theta_max"])
            expected_grid = sorted(k[2] for k in points if k[0] == name and k[3] == beta and k[1] == int(size) and low <= k[2] <= high)
            if "L_small" in row:
                other = int(row["L_small"] if int(size) == int(row["L_large"]) else row["L_large"])
                expected_grid = [x for x in expected_grid if (name, other, x, beta) in points]
            assert min(grid) == low and max(grid) == high
        assert grid == expected_grid and int(row["n"]) == len(grid)
        assert int(row["parameter_count"]) == 2+len(MODELS[model])
        assert int(row["residual_dof"]) == len(grid)-2-len(MODELS[model])
        identity = name, beta, tuple(grid), model, size
        if identity not in cache:
            xx = [mp.mpf(str(x)) for x in grid]
            matrix = mp.matrix([[mp.log(x), mp.mpf(1)]+[x**(-power) for power in MODELS[model]] for x in xx])
            yy = mp.matrix([mp.mpf(points[name, int(x) if domain == "limit" else int(size),
                                         math.inf if domain == "limit" else x, beta]["D"]) for x in grid])
            parameters, _ = mp.qr_solve(matrix, yy)
            residual = yy-matrix*parameters
            cache[identity] = parameters, mp.sqrt(sum(x*x for x in residual)/len(grid))
        parameters, rmse = cache[identity]
        targets = [("c_small", parameters[0])] if small else list(zip(["c", "b"]+[f"inv{p}" for p in MODELS[model]], parameters))
        for field, value in targets:
            close("log_coefficient" if field in ("c", "c_small") else "other_parameter", value, row[field],
                  absolute=1e-8 if field in ("c", "c_small") else 1e-6)
        if not small: close("fit_rmse", rmse, row["rmse"], absolute=1e-9)
        fit_counts[domain+("_small_pair" if small else "")] += 1

    for row in tables["limit_fits"]: check_fit(row, "limit")
    for row in tables["time_individual_fits"]: check_fit(row, "time", row["L"])
    for row in tables["time_pair_fits"]:
        check_fit(row, "time", row["L_large"])
        check_fit(row, "time", row["L_small"], small=True)
        assert truth(row["scale_screen_both"]) == (float(row["theta_min"]) >= plan["theta_min_screen"]
            and float(row["theta_max"])/int(row["L_small"]) <= plan["theta_over_L_max_screen"])
    # Verify every expected limit/window/model fit is present, including small pairs.
    for domain, table in (("limit", "limit_fits"), ("time", "time_individual_fits"), ("pair", "time_pair_fits")):
        expected = set()
        for name, beta in itertools.product(metadata, plan["betas"]):
            config = metadata[name]["limit_fit"]
            sizes = sorted({k[1] for k in points if k[0] == name and k[3] == beta and math.isfinite(k[2])})
            if domain == "limit":
                cases = ((low, config["Lmax"], model) for low, model in itertools.product(config["lower_cuts"], ("log_inv1", "log_inv1_inv2")))
            else:
                cases = itertools.product(sizes if domain == "time" else itertools.combinations(sizes, 2), plan["windows"], MODELS)
            for first, second, model in cases:
                if domain == "limit":
                    low, high = first, second
                    grid = sorted(k[1] for k in points if k[0] == name and k[3] == beta and math.isinf(k[2]) and low <= k[1] <= high)
                    identity = name, beta, low, high, model
                else:
                    low, high = second
                    size = first if domain == "time" else first[0]
                    grid = sorted(k[2] for k in points if k[0] == name and k[3] == beta and k[1] == size and low <= k[2] <= high)
                    if domain == "pair": grid = [x for x in grid if (name, first[1], x, beta) in points]
                    identity = (name, beta, first, low, high, model) if domain == "time" else (name, beta, *first, low, high, model)
                if len(grid) >= len(MODELS[model])+4 and (domain == "limit" or (grid[0] == low and grid[-1] == high)):
                    expected.add(identity)
        actual = []
        for row in tables[table]:
            common = (row["pattern"], float(row["beta"]))
            if domain == "limit": identity = common+(int(row["requested_Lmin"]), int(row["requested_Lmax"]), row["model"])
            elif domain == "time": identity = common+(int(row["L"]), float(row["theta_min"]), float(row["theta_max"]), row["model"])
            else: identity = common+(int(row["L_small"]), int(row["L_large"]), float(row["theta_min"]), float(row["theta_max"]), row["model"])
            actual.append(identity)
        assert len(actual) == len(set(actual)) and set(actual) == expected, (domain, "Missing/duplicate fit")
    limit_index = {(r["pattern"], float(r["beta"]), int(r["requested_Lmin"]), r["model"]): r for r in tables["limit_fits"]}
    pair_index = {(r["pattern"], float(r["beta"]), int(r["L_small"]), int(r["L_large"]), r["window"], r["model"]): r for r in tables["time_pair_fits"]}
    primary_ids = set()
    for row in tables["primary_parameters"]:
        name, beta = row["pattern"], float(row["beta"])
        assert (name, beta) not in primary_ids
        primary_ids.add((name, beta))
        meta, primary = metadata[name], plan["primary"]
        config = meta["limit_fit"]
        assert int(row["limit_Lmin"]) == config["Lmin"] and int(row["limit_Lmax"]) == config["Lmax"]
        assert row["limit_model"] == "log_inv1"
        if row.get("c_limit_fit"):
            anchor = limit_index[name, beta, config["Lmin"], "log_inv1"]
            assert truth(anchor["extent_complete"]) and float(anchor["c"]) == float(row["c_limit_fit"])
            assert anchor["grid"] == row["limit_grid"]
        else: assert snap.incomplete
        if row.get("c_time_fit"):
            assert [float(x) for x in row["primary_window"].split("-")] == primary["window"]
            assert int(row["primary_L"]) == primary["L"] and int(row["comparison_L"]) == primary["comparison_L"]
            assert row["primary_model"] == primary["model"]
            anchor = pair_index[name, beta, primary["comparison_L"], primary["L"], row["primary_window"], primary["model"]]
            assert float(anchor["c"]) == float(row["c_time_fit"]) == float(row["c_time_diagnostic"])
            assert row["grid"] == anchor["grid"]
        else: assert snap.incomplete
        assert truth(row["ground_state_local_checks_pass"]) == (truth(meta["ground_reachable"]) and truth(row["numeric_diagnostic_pass"]))
    assert primary_ids == set(itertools.product(metadata, plan["betas"]))
    for row in tables["local_log_slopes"]:
        name, beta = row["pattern"], float(row["beta"])
        limit = row["domain"] == "limit"
        left, right = float(row["x_left"]), float(row["x_right"])
        keys = [(name, int(x) if limit else int(row["L"]), math.inf if limit else x, beta) for x in (left, right)]
        yy = [mp.mpf(points[k]["D"]) for k in keys]
        close("local_log_slope", (yy[1]-yy[0])/mp.log(mp.mpf(str(right))/mp.mpf(str(left))), row["local_log_slope"])
    return dict(status="PASS", precision_digits=80, summary_sha256=sha(summary_raw),
        analysis_script_sha256=summary["script_sha256"], fit_counts=dict(fit_counts), primary_rows=len(primary_ids),
        max_numerical_errors=dict(maxima), unique_numerical_fits=len(cache),
        table_sha256={name: sha(snap.read(OUT/f"{name}.csv")) for name in table_names},
        method="Independent mpmath 80-digit rectangular QR from raw decimal D values; no production fit helper"), receipt["files"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    snap = Snapshot(args.allow_incomplete)
    result = dict(stage="development" if args.allow_incomplete else "final",
                  generated_utc=datetime.now(timezone.utc).isoformat(), script_sha256=sha(snap.read(Path(__file__))))
    try:
        snap.read(ROOT/"validate_irregular_followup.py")
        plan_raw = snap.read(OUT/"plan.json")
        plan = json.loads(plan_raw.decode("utf-8-sig"))
        selected, data = audit_data(snap, plan)
        result.update(data)
        analysis, bound_outputs = audit_analysis(snap, plan, selected)
        result.update(analysis_validation=analysis, outputs_hashes=bound_outputs,
                      summary_sha256=analysis.get("summary_sha256"), table_sha256=analysis.get("table_sha256", {}),
                      plan_sha256=sha(plan_raw), source_sha256=snap.hashes(),
                      incomplete_tail_bytes=snap.truncated, changed_during_validation=snap.changed())
        result.update(files={path.replace("\\", "/"): digest for path, digest in snap.hashes().items()},
                      precision_digits=analysis.get("precision_digits"),
                      fit_count=sum(analysis.get("fit_counts", {}).values()),
                      max_fit_error=analysis.get("max_numerical_errors", {}).get("log_coefficient"),
                      max_fit_error_scope="Maximum absolute error of the logarithmic coefficient against independent 80-digit QR",
                      primary_rows=analysis.get("primary_rows", 0))
        assert args.allow_incomplete or not result["changed_during_validation"], "Files changed during final audit"
        result["status"] = "PASS_DEVELOPMENT" if args.allow_incomplete else "PASS"
        result["scope"] = ("Frozen plan and complete observations; all four class-2 masks, cut compatibility, D identity, "
            "numerical orbital/entropy diagnostics, physical GS-accessibility flags including blocked accessible excited-state limits; "
            "per-row ledger and complete run/code/plan provenance; independent small real-space reference for all 48 states, "
            "with strict spectral-support Inf construction; 80-digit limit/time fits, primary anchors and raw secants. "
            "Blocked states are never certified as ground states or ground-state corner-coefficient estimates. "
            "Passing numerical checks is not proof of asymptotic logarithmic scaling.")
    except (AssertionError, KeyError, ValueError, IndexError, OSError) as error:
        result.update(status="FAIL", error=repr(error), changed_during_validation=snap.changed(), source_sha256=snap.hashes())
    OUT.mkdir(exist_ok=True)
    (OUT/"validation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({field: result[field] for field in ("status", "stage", "required_rows", "missing_required_rows", "error") if field in result}))
    if result["status"] == "FAIL": raise SystemExit(1)


if __name__ == "__main__":
    main()
