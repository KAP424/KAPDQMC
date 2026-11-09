"""Independent data/provenance and 80-digit fit audit for the irregular follow-up.

Default mode requires complete planned data and a current final analysis. Use
--allow-incomplete only during development; it can never produce final PASS.
No production fit helper is imported and no simulation input is modified.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import re
import sys
import tomllib

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "analysis_irregular_followup"
OLD_DATA_SHA = "7321ec21400690f1b353d729192a90691b347df67f43f201b571e948978b1a50"
OLD_AUDIT_SHA = "a4886015ad701a64aecedf73de900b679b039e2eb508abd283e5eb0e8312092f"
MODELS = {"log_inv1": (1,), "log_inv1_inv2": (1, 2), "log_inv1_inv2_inv3": (1, 2, 3)}


def sha(content):
    return hashlib.sha256(content).hexdigest()


def key(row):
    return row["pattern"], int(row["L"]), float(row["theta"]), round(float(row["beta"]), 10)


def json_key(identity):
    p, size, theta, beta = identity
    return [p, size, "Inf" if math.isinf(theta) else theta, beta]


def truth(value):
    assert str(value).lower() in ("true", "false"), value
    return str(value).lower() == "true"


class Snapshot:
    def __init__(self, incomplete):
        self.incomplete = incomplete
        self.raw = {}
        self.truncated = {}

    def read(self, path):
        path = Path(path).resolve()
        if path not in self.raw:
            self.raw[path] = path.read_bytes()
        return self.raw[path]

    def table(self, path, delimiter=","):
        raw = self.read(path)
        end = raw.rfind(b"\n") + 1
        if end != len(raw):
            assert self.incomplete, (str(path), "incomplete final line")
            self.truncated[str(path)] = len(raw)-end
        prefix = raw[:end]
        assert prefix, (str(path), "missing complete header")
        lines = prefix.decode("utf-8-sig").splitlines()
        rows = list(csv.DictReader(io.StringIO("\n".join(lines)), delimiter=delimiter))
        assert all(None not in row and None not in row.values() for row in rows), path
        return prefix, rows, lines

    def changed(self):
        return [str(p) for p, content in self.raw.items() if not p.exists() or p.read_bytes() != content]

    def hashes(self):
        return {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha(content)
                for p, content in self.raw.items()}


def audit_data(snap, plan):
    metadata = plan["pattern_metadata"]
    assert set(metadata) == set(plan["patterns"]) == {"pure02_block4", "pure02_irregular4", "near02_irregular3"}
    old_audit_raw = snap.read(ROOT / "analysis_occupancy_classes/data_validation.json")
    assert sha(old_audit_raw) == OLD_AUDIT_SHA, "Original final data audit changed"
    old_audit = json.loads(old_audit_raw)
    assert old_audit["status"] == "PASS" and old_audit["stage"] == "final"
    assert sha(snap.read(DATA / "occupancy_classes_observations.csv")) == OLD_DATA_SHA, "Original data changed"
    assert old_audit["inputs"]["data/occupancy_classes_observations.csv"]["file_sha256"] == OLD_DATA_SHA
    old_declarations = {
        name: (family, int(m), bits, int(cls))
        for name, family, m, bits, cls in re.findall(
            r'\("([^"]+)",\s*"([^"]+)",\s*(\d+),\s*"([01]+)",\s*(\d+)\)',
            snap.read(ROOT/"occupancy_classes_patterns.jl").decode())}
    new_definition = snap.read(ROOT/"irregular_m3_patterns.jl").decode()
    new_bits = re.search(r'bits\s*=\s*"([01]+)"', new_definition).group(1)
    classes = []
    for name, meta in metadata.items():
        m, bits = meta["m"], meta["occupied_bits"]
        assert len(bits) == 2*m*m and bits.count("1") == m*m and set(bits) <= {"0", "1"}
        counts = Counter(int(bits[i])+int(bits[i+1]) for i in range(0, len(bits), 2))
        cls = 1 if counts[1] == m*m else 2 if counts[1] == 0 else 3
        assert cls == meta["occupancy_class"] and counts[0] == counts[2]
        if name in old_declarations:
            assert old_declarations[name] == (meta["family"], m, bits, cls)
        else:
            assert name == "near02_irregular3" and m == 3 and bits == new_bits
            assert (counts[0], counts[1], counts[2]) == (4, 1, 4)
            assert 'name="near02_irregular3"' in new_definition
        classes.append(dict(pattern=name, m=m, n0=counts[0], n1=counts[1], n2=counts[2], occupancy_class=cls))

    physics = {}
    for stem in ("occupancy_classes_validation", "irregular_m3_validation", "irregular_ground_validation"):
        receipt_raw = snap.read(DATA/f"{stem}.toml")
        proof = tomllib.loads(receipt_raw.decode())
        assert proof["status"] == "PASS"
        assert sha(snap.read(DATA/f"{stem}.csv")) == proof["csv_sha256"]
        for filename, digest in proof["code_sha256"].items():
            assert sha(snap.read(ROOT/filename)) == digest, (filename, "changed since physical validation")
        physics[stem] = proof
    m3 = physics["irregular_m3_validation"]
    assert m3["states"] == 9 and m3["occupied_bits"] == new_bits and m3["L"] == 9
    assert m3["max_covariance_error"] < 1e-9 and m3["max_entropy_error"] < 1e-9
    assert m3["dense_theta64_128_projector_difference"] < 1e-10
    assert abs(m3["dense_ground_excess"]) < 1e-9
    _, physical_rows, _ = snap.table(DATA/"irregular_m3_validation.csv")
    assert {(float(r["theta"]), float(r["beta"])) for r in physical_rows} == set(itertools.product((0., 4., math.inf), (0., 1., 2.)))
    kernel_sha = m3["code_sha256"]["fermion_experiment.jl"]
    assert physics["occupancy_classes_validation"]["code_sha256"]["fermion_experiment.jl"] == kernel_sha

    # The original m=4 states need an independent limiting-state comparison too:
    # matching energy alone would not identify their occupied zero-mode subspace.
    m4 = physics["irregular_ground_validation"]
    assert m4["code_sha256"]["fermion_experiment.jl"] == kernel_sha
    assert m4["states"] == 6 and m4["L"] == 12 and m4["zero_mode_dimension"] == 4
    assert m4["step"] == .25 and m4["failures"] == []
    _, m4_rows, _ = snap.table(DATA/"irregular_ground_validation.csv")
    m4_names = {"pure02_block4", "pure02_irregular4"}
    assert len(m4_rows) == 6
    assert {(r["pattern"], float(r["theta"]), float(r["beta"])) for r in m4_rows} == set(
        itertools.product(m4_names, (math.inf,), (0., 1., 2.)))
    maxima_fields = {
        "max_covariance_error": "covariance_error",
        "max_covariance_element_error": "covariance_max_element",
        "max_entropy_error": "entropy_error",
        "max_D_error": "D_error",
        "max_zero_sector_error": "premeasurement_zero_sector_error",
    }
    for recorded, column in maxima_fields.items():
        maximum = max(float(r[column]) for r in m4_rows)
        assert math.isfinite(maximum) and 0 <= maximum < 1e-9
        assert maximum == m4[recorded], (recorded, "m=4 receipt/CSV maximum mismatch")
    pattern_checks = {p["pattern"]: p for p in m4["patterns"]}
    assert len(m4["patterns"]) == 2 and set(pattern_checks) == m4_names
    m4_pattern_summaries = []
    for name, check in pattern_checks.items():
        assert check["occupied_bits"] == metadata[name]["occupied_bits"]
        assert check["reference_theta"] in (128., 256.) and check["reference_convergence"] < 1e-10
        assert abs(check["dense_ground_excess"]) < 1e-9 and check["dense_ground_residual"] < 1e-9
        assert abs(check["premeasurement_zero_sector_particles"]-2) < 1e-10
        assert check["premeasurement_zero_sector_error"] < 1e-9
        m4_pattern_summaries.append({field: check[field] for field in (
            "pattern", "occupied_bits", "reference_theta", "reference_convergence",
            "dense_ground_excess", "dense_ground_residual", "premeasurement_zero_sector_error",
            "premeasurement_zero_sector_particles")})
    for row in m4_rows:
        assert int(row["L"]) == 12 and int(row["m"]) == 4
        check = pattern_checks[row["pattern"]]
        assert float(row["reference_theta"]) == check["reference_theta"]
        assert float(row["dense_reference_convergence"]) == check["reference_convergence"]
        assert float(row["dense_ground_excess"]) == check["dense_ground_excess"]
        assert float(row["dense_residual"]) == check["dense_ground_residual"]
        assert float(row["premeasurement_zero_sector_error"]) == check["premeasurement_zero_sector_error"]
        assert float(row["premeasurement_zero_sector_particles"]) == check["premeasurement_zero_sector_particles"]
        assert float(row["entropy_error"]) == max(abs(float(row[f])-float(row["dense_"+f])) for f in ("S_A", "S_C", "S_B"))
        assert float(row["D_error"]) == abs(float(row["D"])-float(row["dense_D"]))
        for prefix in ("", "dense_"):
            d = (2*float(row[prefix+"S_A"])+float(row[prefix+"S_C"]))/3-float(row[prefix+"S_B"])
            assert abs(d-float(row[prefix+"D"])) < 1e-12
    m4_summary = dict(status="PASS", L=12, states=6, zero_mode_dimension=4,
        **{field: m4[field] for field in maxima_fields}, patterns=m4_pattern_summaries,
        receipt_sha256=sha(snap.read(DATA/"irregular_ground_validation.toml")),
        csv_sha256=m4["csv_sha256"], code_sha256=m4["code_sha256"], scope=m4["scope"])

    required = set()
    for grid in plan["required_equilibrium_grids"]+plan["required_finite_grids"]:
        required.update(itertools.product(grid["patterns"], grid["L"], map(float, grid["theta"]), grid["beta"]))
    _, reachability, _ = snap.table(DATA/"irregular_m3_reachability.csv")
    assert {int(r["L"]) for r in reachability} >= {k[1] for k in required if k[0] == "near02_irregular3"}
    for row in reachability:
        assert row["occupied_bits"] == new_bits and truth(row["ground_reachable"])
        assert row["pattern_sha256"] == sha(snap.read(ROOT/"irregular_m3_patterns.jl"))
        assert row["kernel_sha256"] == kernel_sha
        assert int(row["negative_rank_deficiency"]) == int(row["nonpositive_rank_deficiency"]) == 0
        assert float(row["min_negative_sv"]) > 1e-8 and float(row["min_nonpositive_sv"]) > 1e-8
        assert abs(float(row["excess_energy"])) < 1e-8

    selected, input_receipts, duplicate_rows, run_receipts = {}, {}, [], []
    maxima, pending = {}, []
    verified_code = set()

    def bounded(field, value, bound, identity):
        value = abs(float(value))
        assert math.isfinite(value) and value < bound, (identity, field, value, bound)
        maxima[field] = max(maxima.get(field, 0.), value)

    for relative in plan["inputs"]:
        path = ROOT/relative
        if not path.exists():
            assert snap.incomplete, (relative, "missing planned source")
            pending.append(relative)
            continue
        content, records, lines = snap.table(path)
        ledger_path = path.with_name(path.stem+"_provenance.tsv")
        _, ledger, _ = snap.table(ledger_path, "\t")
        if len(records) != len(ledger):
            assert snap.incomplete, (relative, "CSV/ledger lengths differ")
            pending.append(relative+": unpaired CSV/ledger tail")
        paired = min(len(records), len(ledger))
        records = records[:paired]
        assert len({key(row) for row in records}) == len(records), (relative, "duplicate keys within source")
        run_groups = defaultdict(list)
        for number, (row, entry) in enumerate(zip(records, ledger[:paired]), 2):
            identity = (relative, number)
            assert int(entry["output_row"]) == number and entry["csv_row"] == lines[number-1], identity
            assert entry["row_sha256"] == sha(lines[number-1].encode()), identity
            assert all(entry[f] == row[f] for f in ("run_id", "source_kind", "source_path", "source_row", "source_sha256")), identity
            assert row["source_kind"] == "computed" and row["entropy_method"] == "real_cholesky", identity
            run_groups[row["run_id"]].append((number, row))
            name, size, theta, beta = key(row)
            if name not in metadata:
                continue
            meta = metadata[name]
            assert (int(row["m"]), row["family"], row["occupied_bits"]) == (meta["m"], meta["family"], meta["occupied_bits"]), identity
            assert size >= 3 and size % (3*meta["m"]) == 0, identity
            assert theta >= 0 and not math.isnan(theta) and beta >= 0 and math.isfinite(beta), identity
            assert truth(row["premeasurement_ground"]), identity
            assert float(row["rank_tol"]) == 1e-10, identity
            for field in ("x0", "y0", "cut_x0", "cut_y0"):
                if row.get(field):
                    assert int(row[field]) == 1, identity
            sa, sc, sb, d = (float(row[f]) for f in ("S_A", "S_C", "S_B", "D"))
            assert all(math.isfinite(x) and x >= -1e-8 for x in (sa, sc, sb)), identity
            bounded("D_identity", d-((2*sa+sc)/3-sb), 1e-9, identity)
            bounded("delta_identity", float(row["delta"])-(sa-sb), 1e-9, identity)
            for field, bound in (("premeasurement_excess", 1e-8), ("premeasurement_residual", 1e-9),
                    ("accessible_excess", 1e-8), ("projector_error", 1e-10), ("reconstruction_error", 1e-9),
                    ("rectangle_imag_error", 2e-10), ("rectangle_real_error_bound", 1e-9)):
                bounded(field, row[field], bound, identity)
            if math.isinf(theta) and beta == 0:
                bounded("unmeasured_ground_excess", row["excess_energy"], 1e-8, identity)
                bounded("unmeasured_ground_residual", row["residual"], 1e-9, identity)
            if math.isfinite(theta) and beta == 0 and 4 <= theta <= 8 and size >= 144:
                assert float(row["residual"]) > 1e-8, (identity, "finite theta replaced by ground state")
            if beta not in plan["betas"] or theta == 0:
                continue
            k = key(row)
            if k in selected:
                old_row = selected[k][0]
                difference = max(abs(float(row[f])-float(old_row[f])) for f in ("S_A", "S_C", "S_B", "D"))
                assert difference <= 1e-9, (identity, "conflicting duplicate", difference)
                duplicate_rows.append(dict(key=json_key(k), difference=difference))
            selected[k] = (row, str(path.resolve()), number)
        for rid, run_rows in run_groups.items():
            candidates = list(DATA.glob(f"*_run_{rid}.toml"))
            assert len(candidates) == 1, (rid, "missing or ambiguous manifest")
            manifest_path = candidates[0]
            manifest_raw = snap.read(manifest_path)
            manifest = tomllib.loads(manifest_raw.decode())
            assert manifest["run_id"] == rid and Path(manifest["output"]).resolve() == path.resolve()
            if manifest["status"] != "complete":
                assert snap.incomplete and manifest["status"] in ("running", "failed", "interrupted"), (rid, manifest["status"])
                pending.append(f"{rid}: {manifest['status']}")
            hashes = manifest["code_sha256"]
            assert hashes["fermion_experiment.jl"] == kernel_sha
            runners = {name: digest for name, digest in hashes.items() if name in
                       ("irregular_followup_experiments.jl", "occupancy_classes_experiments.jl")}
            assert len(runners) == 1
            runner, runner_hash = next(iter(runners.items()))
            for filename, digest in hashes.items():
                stored = DATA/"provenance/code"/digest/filename
                assert sha(snap.read(stored)) == digest, (rid, filename, "snapshot hash")
                verified_code.add((filename, digest))
            expected = set(itertools.product(manifest["patterns"], manifest["L"], map(float, manifest["theta"]), manifest["beta"]))
            for _, row in run_rows:
                assert row["code_sha256"] == runner_hash and key(row) in expected
            if manifest["status"] == "complete":
                assert len(run_rows) == manifest["new_rows"]
                final_row = max(number for number, _ in run_rows)
                assert sha(b"".join(content.splitlines(keepends=True)[:final_row])) == manifest["output_sha256"]
                grounds = {(g["pattern"], int(g["L"])): g for g in manifest["ground_checks"]}
                assert {(r["pattern"], int(r["L"])) for _, r in run_rows} <= set(grounds)
                for ground in grounds.values():
                    bounded("manifest_ground_excess", ground.get("excess", ground.get("excess_energy")), 1e-8, rid)
                    bounded("manifest_ground_residual", ground["residual"], 1e-9, rid)
            run_receipts.append(dict(path=str(manifest_path.relative_to(ROOT)), sha256=sha(manifest_raw),
                                     status=manifest["status"], rows=len(run_rows), runner=runner))
        input_receipts[relative] = dict(sha256=sha(snap.read(path)), rows=len(records),
            ledger_sha256=sha(snap.read(ledger_path)), paired_rows=paired)
    missing = required-set(selected)
    assert snap.incomplete or not missing, ("missing required plan keys", len(missing))
    return selected, dict(required_rows=len(required), available_required_rows=len(required-missing),
        missing_required_rows=len(missing), missing_keys=[json_key(k) for k in sorted(missing)],
        pattern_metadata=classes, input_receipts=input_receipts, audited_runs=run_receipts,
        code_snapshots_checked=len(verified_code), consistent_duplicate_rows=duplicate_rows,
        numerical_maxima=maxima, pending_sources_or_runs=pending,
        old_data_sha256=OLD_DATA_SHA, old_audit_sha256=OLD_AUDIT_SHA,
        m3_physical_validation_sha256=sha(snap.read(DATA/"irregular_m3_validation.toml")),
        m4_ground_validation=m4_summary,
        m4_ground_validation_sha256=m4_summary["receipt_sha256"])


def load_mpmath():
    try:
        import mpmath
    except ImportError:
        sys.path.insert(0, str(ROOT/"analysis_beta_patterns/audit_dependencies"))
        import mpmath
    return mpmath


def audit_analysis(snap, plan, selected):
    """Use 80-digit rectangular QR directly on observed decimal strings."""
    summary_raw = snap.read(OUT/"summary.json")
    summary = json.loads(summary_raw)
    stale = []
    if summary["script_sha256"] != sha(snap.read(ROOT/"analyze_irregular_followup.py")):
        stale.append("analysis script changed")
    if summary["plan_sha256"] != sha(snap.read(OUT/"plan.json")):
        stale.append("plan changed")
    for source in summary["sources"]:
        path = Path(source["path"])
        if source.get("missing"):
            if path.exists():
                stale.append(str(path)+": previously missing")
        elif not path.exists() or sha(snap.read(path)) != source["sha256"]:
            stale.append(str(path)+": data changed")
    for dependency in summary["dependencies"]:
        if sha(snap.read(dependency["path"])) != dependency["sha256"]:
            stale.append(dependency["path"]+": dependency changed")
    if snap.incomplete and stale:
        return dict(status="SKIPPED_STALE_DEVELOPMENT_ANALYSIS", stale=stale, summary_sha256=sha(summary_raw)), {}
    assert not stale, stale
    if not snap.incomplete:
        assert summary["stage"] == "final" and summary["primary_complete"] and summary["missing_required_keys"] == 0
        assert not summary["source_changes_during_run"] and not summary["dependency_changes_during_run"] and not summary["plan_changed_during_run"]
    receipt = json.loads(snap.read(OUT/"receipt.json"))
    assert receipt["script_sha256"] == summary["script_sha256"] and receipt["plan_sha256"] == summary["plan_sha256"]
    if not snap.incomplete:
        assert receipt["status"] == "COMPLETE" and receipt["stage"] == "final"
    output_hashes = {}
    for item in receipt["outputs"]:
        assert sha(snap.read(ROOT/item["path"])) == item["sha256"], (item["path"], "analysis output changed")
        output_hashes[item["path"]] = item["sha256"]
    table_names = ("observations", "equilibrium_fits", "time_individual_fits", "time_pair_fits",
                   "primary_coefficients", "heldout_extrapolation", "extended_equilibrium_fits", "local_log_slopes")
    tables = {name: snap.table(OUT/f"{name}.csv")[1] for name in table_names}
    points = {key(row): row for row in tables["observations"]}
    assert len(points) == len(tables["observations"]) and set(points) == set(selected), "Merged observation coverage mismatch"
    for identity, row in points.items():
        original, path, number = selected[identity]
        assert int(row["source_row"]) == number and (ROOT/row["source_file"]).resolve() == Path(path)
        assert row["source_sha256"] == sha(snap.read(path))
        assert row["occupied_bits"] == original["occupied_bits"] and int(row["x0"]) == int(row["y0"]) == 1
        assert all(float(row[f]) == float(original[f]) for f in ("S_A", "S_C", "S_B", "D"))

    mp = load_mpmath()
    mp.mp.dps = 80
    max_error, fit_counts, cache = defaultdict(float), Counter(), {}

    def close(field, expected, actual, abs_tol=1e-8, rel_tol=1e-8):
        expected, actual = float(expected), float(actual)
        error = abs(expected-actual)
        assert math.isfinite(actual) and error <= abs_tol+rel_tol*abs(expected), (field, expected, actual, error)
        max_error[field] = max(max_error[field], error)

    def reconstruct(pattern, beta, grid, model, size=None):
        identity = pattern, beta, tuple(grid), model, size
        if identity in cache:
            return cache[identity]
        powers = MODELS[model]
        assert len(grid) >= len(powers)+4 and grid == sorted(set(grid))
        xx = [mp.mpf(str(x)) for x in grid]
        matrix = mp.matrix([[mp.log(x), mp.mpf(1)]+[x**(-power) for power in powers] for x in xx])
        yy = mp.matrix([mp.mpf(points[(pattern, int(x) if size is None else int(size),
                                      math.inf if size is None else x, beta)]["D"]) for x in grid])
        parameters, _ = mp.qr_solve(matrix, yy)
        residuals = yy-matrix*parameters
        rmse = mp.sqrt(sum(value**2 for value in residuals)/len(grid))
        cache[identity] = (parameters, rmse)
        return parameters, rmse

    def check_fit(row, domain, size=None, small=False):
        grid = [float(v) for v in row["grid"].split(",")]
        model = row["model"]
        name, beta = row["pattern"], float(row["beta"])
        if domain == "equilibrium":
            low, high = float(row["requested_Lmin"]), float(row["requested_Lmax"])
            expected_grid = sorted(k[1] for k in points if k[0] == name and k[3] == beta
                                   and math.isinf(k[2]) and low <= k[1] <= high)
        else:
            low, high = float(row["theta_min"]), float(row["theta_max"])
            expected_grid = sorted(k[2] for k in points if k[0] == name and k[3] == beta
                                   and k[1] == int(size) and low <= k[2] <= high)
            if "L_small" in row:
                other_size = int(row["L_small"] if int(size) == int(row["L_large"]) else row["L_large"])
                expected_grid = [x for x in expected_grid if (name, other_size, x, beta) in points]
        assert grid == expected_grid, (name, beta, domain, "fit omitted or inserted an observed point")
        pars, rmse = reconstruct(row["pattern"], float(row["beta"]), grid, model, size)
        assert int(row["n"]) == len(grid)
        assert int(row["parameter_count"]) == 2+len(MODELS[model])
        assert int(row["residual_dof"]) == len(grid)-2-len(MODELS[model])
        assert float(row["x_min"]) == min(grid) and float(row["x_max"]) == max(grid)
        if domain == "equilibrium":
            assert min(grid) >= float(row["requested_Lmin"]) and max(grid) <= float(row["requested_Lmax"])
            assert truth(row["extent_complete"]) == (min(grid) == low and max(grid) == high)
        else:
            assert min(grid) == float(row["theta_min"]) and max(grid) == float(row["theta_max"])
        targets = [("c_small", pars[0])] if small else list(zip(["c", "b"]+[f"inv{power}" for power in MODELS[model]], pars))
        for field, expected in targets:
            close("log_coefficient" if field in ("c", "c_small") else "other_parameter", expected, row[field],
                  abs_tol=1e-8 if field in ("c", "c_small") else 1e-6)
        if not small:
            close("fit_rmse", rmse, row["rmse"], abs_tol=1e-9)
        fit_counts[domain+("_small_pair" if small else "")] += 1
        return pars, rmse

    for table in ("equilibrium_fits", "extended_equilibrium_fits"):
        for row in tables[table]:
            check_fit(row, "equilibrium")
    for row in tables["time_individual_fits"]:
        check_fit(row, "time", row["L"])
    for row in tables["time_pair_fits"]:
        check_fit(row, "time", row["L_large"])
        small, _ = check_fit(row, "time", row["L_small"], small=True)
        large, _ = reconstruct(row["pattern"], float(row["beta"]), [float(v) for v in row["grid"].split(",")], row["model"], row["L_large"])
        close("size_drift", abs((large[0]-small[0])/large[0]), row["size_drift"])
        assert truth(row["size_sign_changed"]) == (large[0]*small[0] <= 0)
        assert truth(row["scale_screen_both"]) == (float(row["theta_min"]) >= plan["theta_min_screen"]
                    and float(row["theta_max"])/int(row["L_small"]) <= plan["theta_over_L_max_screen"])

    # Check the declared fit collections are complete, not just numerically valid.
    expected_eq, expected_time, expected_pairs = set(), set(), set()
    for name, beta in itertools.product(plan["patterns"], plan["betas"]):
        config = plan["pattern_metadata"][name]["equilibrium"]
        for low, model in itertools.product(config["lower_cuts"], ("log_inv1", "log_inv1_inv2")):
            grid = [k[1] for k in points if k[0] == name and k[3] == beta and math.isinf(k[2])
                    and low <= k[1] <= config["Lmax"]]
            if len(grid) >= len(MODELS[model])+4:
                expected_eq.add((name, beta, low, config["Lmax"], model))
        sizes = sorted({k[1] for k in points if k[0] == name and k[3] == beta and math.isfinite(k[2])})
        for size, (low, high), model in itertools.product(sizes, plan["windows"], MODELS):
            grid = sorted(k[2] for k in points if k[0] == name and k[3] == beta and k[1] == size and low <= k[2] <= high)
            if len(grid) >= len(MODELS[model])+4 and grid[0] == low and grid[-1] == high:
                expected_time.add((name, beta, size, low, high, model))
        for (small, large), (low, high), model in itertools.product(itertools.combinations(sizes, 2), plan["windows"], MODELS):
            grid = sorted(k[2] for k in points if k[0] == name and k[3] == beta and k[1] == small
                          and low <= k[2] <= high and (name, large, k[2], beta) in points)
            if len(grid) >= len(MODELS[model])+4 and grid[0] == low and grid[-1] == high:
                expected_pairs.add((name, beta, small, large, low, high, model))
    actual_eq = [(r["pattern"], float(r["beta"]), int(r["requested_Lmin"]), int(r["requested_Lmax"]), r["model"]) for r in tables["equilibrium_fits"]]
    actual_time = [(r["pattern"], float(r["beta"]), int(r["L"]), float(r["theta_min"]), float(r["theta_max"]), r["model"]) for r in tables["time_individual_fits"]]
    actual_pairs = [(r["pattern"], float(r["beta"]), int(r["L_small"]), int(r["L_large"]), float(r["theta_min"]), float(r["theta_max"]), r["model"]) for r in tables["time_pair_fits"]]
    for actual, expected in ((actual_eq, expected_eq), (actual_time, expected_time), (actual_pairs, expected_pairs)):
        assert len(actual) == len(set(actual)) and set(actual) == expected, "Incomplete or duplicate fit collection"

    primary_identities = set()
    eq_index = {(r["pattern"], float(r["beta"]), int(r["requested_Lmin"]), r["model"]): r for r in tables["equilibrium_fits"]}
    pair_index = {(r["pattern"], float(r["beta"]), int(r["L_small"]), int(r["L_large"]), r["window"], r["model"]): r for r in tables["time_pair_fits"]}
    for row in tables["primary_coefficients"]:
        identity = row["pattern"], float(row["beta"])
        assert identity not in primary_identities
        primary_identities.add(identity)
        config = plan["pattern_metadata"][identity[0]]["equilibrium"]
        assert int(row["eq_Lmin"]) == config["Lmin"] and int(row["eq_Lmax"]) == config["Lmax"]
        assert row["eq_model"] == plan["equilibrium_primary"]["model"]
        if row.get("c_eq"):
            anchor = eq_index[identity+(config["Lmin"], row["eq_model"])]
            assert truth(anchor["extent_complete"]) and float(anchor["c"]) == float(row["c_eq"])
            assert anchor["grid"] == row["eq_grid"]
        else:
            assert snap.incomplete, (identity, "missing equilibrium primary")
        if row.get("c_neq"):
            primary = plan["primary"]
            assert (int(row["primary_L"]), int(row["comparison_L"]), row["primary_model"]) == (primary["L"], primary["comparison_L"], primary["model"])
            assert [float(v) for v in row["primary_window"].split("-")] == primary["window"]
            anchor = pair_index[identity+(primary["comparison_L"], primary["L"], row["primary_window"], primary["model"])]
            assert float(anchor["c"]) == float(row["c_neq"]) == float(row["c_neq_diagnostic"])
            assert row["grid"] == anchor["grid"]
        else:
            assert snap.incomplete, (identity, "missing time primary")
    assert primary_identities == set(itertools.product(plan["patterns"], plan["betas"]))

    expected_extrapolations = set()
    for row in tables["equilibrium_fits"]:
        if truth(row["extent_complete"]):
            for k in points:
                if k[0] == row["pattern"] and k[3] == float(row["beta"]) and math.isinf(k[2]) and k[1] > int(row["requested_Lmax"]):
                    expected_extrapolations.add((k[0], k[3], k[1], int(row["requested_Lmin"]), row["model"]))
    seen_extrapolations = set()
    for row in tables["heldout_extrapolation"]:
        name, beta, heldout, low = row["pattern"], float(row["beta"]), int(row["heldout_L"]), int(row["train_Lmin"])
        identity = name, beta, heldout, low, row["model"]
        assert identity not in seen_extrapolations
        seen_extrapolations.add(identity)
        anchor = eq_index[(name, beta, low, row["model"])]
        assert row["train_grid"] == anchor["grid"] and int(row["train_Lmax"]) == 144 < heldout
        assert truth(row["heldout_not_used_in_fit"])
        grid = [float(v) for v in row["train_grid"].split(",")]
        assert heldout not in grid
        pars, rmse = reconstruct(name, beta, grid, row["model"])
        x = mp.mpf(heldout)
        prediction = pars[0]*mp.log(x)+pars[1]+sum(pars[i+2]*x**(-power) for i, power in enumerate(MODELS[row["model"]]))
        observed = mp.mpf(points[name, heldout, math.inf, beta]["D"])
        close("heldout_c_train", pars[0], row["c_train"])
        close("heldout_observed", observed, row["D_observed"], abs_tol=0, rel_tol=0)
        close("heldout_prediction", prediction, row["D_prediction"])
        close("heldout_residual", observed-prediction, row["residual"])
        close("heldout_abs_residual", abs(observed-prediction), row["abs_residual"])
        close("heldout_train_rmse", rmse, row["train_rmse"], abs_tol=1e-9)
    assert seen_extrapolations == expected_extrapolations
    for row in tables["local_log_slopes"]:
        name, beta = row["pattern"], float(row["beta"])
        left, right = float(row["x_left"]), float(row["x_right"])
        equilibrium = row["domain"] == "equilibrium"
        keys = [(name, int(x) if equilibrium else int(row["L"]), math.inf if equilibrium else x, beta) for x in (left, right)]
        yy = [mp.mpf(points[k]["D"]) for k in keys]
        close("local_log_slope", (yy[1]-yy[0])/mp.log(mp.mpf(str(right))/mp.mpf(str(left))), row["local_log_slope"])
    return dict(status="PASS", precision_digits=80, summary_sha256=sha(summary_raw),
        analysis_script_sha256=summary["script_sha256"], fit_counts=dict(fit_counts), unique_numerical_fits=len(cache),
        primary_rows=len(primary_identities), heldout_predictions=len(seen_extrapolations),
        max_numerical_errors=dict(max_error),
        method="Independent mpmath rectangular QR from unmodified observed decimal strings; no production fit helper imported",
        table_sha256={name: sha(snap.read(OUT/f"{name}.csv")) for name in table_names}), output_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-incomplete", action="store_true", help="Development only; never yields final PASS")
    args = parser.parse_args()
    snap = Snapshot(args.allow_incomplete)
    result = dict(stage="development" if args.allow_incomplete else "final",
                  generated_utc=datetime.now(timezone.utc).isoformat(),
                  script_sha256=sha(snap.read(Path(__file__))))
    try:
        plan_raw = snap.read(OUT/"plan.json")
        plan = json.loads(plan_raw.decode("utf-8-sig"))
        selected, data = audit_data(snap, plan)
        result.update(data)
        analysis, outputs = audit_analysis(snap, plan, selected)
        result.update(analysis_validation=analysis, outputs_hashes=outputs,
                      summary_sha256=analysis.get("summary_sha256"), table_sha256=analysis.get("table_sha256", {}),
                      plan_sha256=sha(plan_raw), source_sha256=snap.hashes(),
                      incomplete_tail_bytes=snap.truncated, changed_during_validation=snap.changed())
        assert args.allow_incomplete or not result["changed_during_validation"], "Files changed during final audit"
        result["status"] = "PASS_DEVELOPMENT" if args.allow_incomplete else "PASS"
        result["scope"] = ("Plan coverage; preserved historical data and audit hashes; occupation metadata, 3m cut compatibility, "
            "D identity and numerical physical diagnostics; per-row CSV/ledger hashes, run manifests and code snapshots; "
            "independent L9 and L12 projector/entropy validation including initial-state-selected zero modes; numerical equilibrium/time "
            "fits, primary anchors, held-out predictions and local secants. This does not certify an asymptotic scaling law "
            "or independently reimplement every finite-window diagnostic gate.")
    except (AssertionError, KeyError, ValueError, IndexError, OSError) as error:
        result.update(status="FAIL", error=repr(error), changed_during_validation=snap.changed(), source_sha256=snap.hashes())
    OUT.mkdir(exist_ok=True)
    (OUT/"validation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "stage", "required_rows", "missing_required_rows", "error") if k in result}, ensure_ascii=False))
    if result["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
