"""Independent final coverage and provenance checks for the frozen beta-zero pilot.

This validates computed/reused observations, not scaling fits or their physics.
The frozen baseline is checked against every recorded source digest, and the
new rows against the exact job file, row ledger, run snapshots and dense test.
"""
from __future__ import annotations

import csv
import argparse
import hashlib
import json
import math
import pathlib
import shutil
import tomllib
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parent
ANALYSIS = ROOT / "analysis_beta0_study"


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def rows(path, delimiter=","):
    raw = pathlib.Path(path).read_text(encoding="utf-8")
    assert raw.endswith("\n"), f"incomplete file {path}"
    return list(csv.DictReader(raw.splitlines(), delimiter=delimiter)), raw.splitlines()[1:]


def key(r):
    return r.get("canonical_pattern", r["pattern"]), int(r["L"]), float(r["theta"]), float(r["beta"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pilot", "final"), default="pilot")
    args = parser.parse_args()
    files = {}

    def bind(path, expected=None):
        path = pathlib.Path(path).resolve()
        value = sha(path)
        if expected is not None:
            assert value == expected, f"changed file {path}"
        try:
            label = path.relative_to(ROOT).as_posix()
        except ValueError:
            label = str(path)
        files[label] = value
        return value

    requests = []
    names = [("study_design.json", "pilot")]
    if args.stage == "final":
        names.append(("confirmation_plan.json", "confirmation"))
    for plan_name, prefix in names:
        plan_path = ANALYSIS / plan_name
        plan_hash = bind(plan_path)
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        for name, digest in plan["sources"].items():
            bind(ROOT / name, digest)
        requests.append(dict(plan=plan, prefix=prefix, plan_hash=plan_hash))
    provenance = json.loads((ANALYSIS / "baseline_provenance.json").read_text(encoding="utf-8"))
    assert provenance["status"] == "PASS"
    bind(ANALYSIS / "baseline_observations.csv", provenance["baseline_sha256"])
    for name, digest in provenance["source_sha256"].items():
        bind(ROOT / name, digest)
    baseline, _ = rows(ANALYSIS / "baseline_observations.csv")
    baseline_keys = {key(r): r for r in baseline}
    assert len(baseline_keys) == len(baseline)
    job_keys, reused_keys, expected, reused = set(), set(), set(), []
    for request in requests:
        plan, prefix = request["plan"], request["prefix"]
        jobs_path = ANALYSIS / f"{prefix}_jobs.csv"
        jobs, _ = rows(jobs_path)
        request["job_hash"] = bind(jobs_path)
        current_jobs = {key(r) for r in jobs}
        assert len(current_jobs) == len(jobs) == plan["new_jobs"]
        reused_rows, _ = rows(ANALYSIS / f"{prefix}_reused.csv")
        current_reused = {key(r) for r in reused_rows}
        assert len(current_reused) == len(reused_rows) == plan["reused_keys"]
        assert not current_jobs & current_reused
        if prefix == "pilot":
            current_expected = {(p, int(L), float(t), 0.) for p in plan["patterns"]
                                for L in plan["pilot_sizes"] for t in plan["pilot_theta"]}
            current_expected |= {(p, int(L), math.inf, 0.) for p in plan["patterns"]
                                 for L in plan["equilibrium_extension_sizes"]}
        else:
            def grid(window):
                a, b, step = (float(window[x]) for x in ("theta_min", "theta_max", "theta_step"))
                return [round(a+i*step, 10) for i in range(round((b-a)/step)+1)]
            current_expected = set()
            for p in plan["patterns"]:
                for L in plan["third_size"]["sizes"]:
                    current_expected |= {(p, int(L), t, 0.) for t in grid(plan["third_size"])}
                for L in (plan["primary"]["L"], plan["primary"]["comparison_L"]):
                    current_expected |= {(p, int(L), t, 0.) for t in grid(plan["primary"])}
                current_expected |= {(p, int(plan["additional_window"]["L"]), t, 0.)
                                     for t in grid(plan["additional_window"])}
        assert len(current_expected) == plan["total_target_keys"]
        assert current_expected == current_jobs | current_reused
        for r in reused_rows:
            previous = baseline_keys[key(r)]
            if "source_id" in r:
                assert previous["source_id"] == r["source_id"]
            assert previous["ground_reachable"].lower() == "true"
        request["job_keys"] = current_jobs
        assert not current_jobs & job_keys
        job_keys |= current_jobs
        reused_keys |= current_reused
        expected |= current_expected
        reused.extend(reused_rows)
    assert not job_keys & reused_keys
    assert len(expected) == len(job_keys) + len(reused_keys)
    by_job_hash = {r["job_hash"]: r for r in requests}

    output = ROOT / "data/beta0_study_observations.csv"
    ledger_path = output.with_name(output.stem + "_provenance.tsv")
    data, raw_lines = rows(output)
    ledger, _ = rows(ledger_path, "\t")
    assert len(data) == len(ledger) == len(job_keys), "pilot incomplete or ledger length mismatch"
    assert {key(r) for r in data} == job_keys
    assert len({key(r) for r in data}) == len(data)
    output_hash = bind(output)
    ledger_hash = bind(ledger_path)
    manifests = {}
    for runid in {r["run_id"] for r in data}:
        path = output.with_name(output.stem + f"_run_{runid}.toml")
        bind(path)
        m = tomllib.loads(path.read_text(encoding="utf-8"))
        assert m["status"] == "complete" and m["run_id"] == runid
        assert pathlib.Path(m["output"]).resolve() == output
        request = by_job_hash[m["job_file_sha256"]]
        assert m["plan_sha256"] == request["plan_hash"]
        assert m["rank_tol"] == 1e-10 and m["beta"] == [0.]
        assert "exact spectral-support" in m["protocol"]
        assert "preserve initial-selected zero modes" in m["protocol"]
        bind(m["jobs_snapshot"], request["job_hash"])
        bind(m["plan_snapshot"], request["plan_hash"])
        for name, digest in m["code_sha256"].items():
            bind(ROOT / name, digest)
            bind(m["code_snapshots"][name], digest)
        run_jobs = {key(r) for r in m["jobs"]}
        run_rows = {key(r) for r in data if r["run_id"] == runid}
        assert run_jobs == run_rows and len(run_rows) == m["new_rows"]
        assert run_jobs <= request["job_keys"]
        m["ground_by_key"] = {(g["pattern"], g["L"]): g for g in m["ground_checks"]}
        manifests[runid] = m
    latest = max(manifests.values(), key=lambda m: m["finished_utc"])
    assert latest["output_sha256"] == output_hash and latest["ledger_sha256"] == ledger_hash
    limits = {"projector_error": 1e-10, "reconstruction_error": 1e-9,
              "premeasurement_excess": 1e-8, "premeasurement_residual": 1e-9,
              "rectangle_imag_error": 2e-10, "rectangle_real_error_bound": 1e-9}
    maxima = {name: 0. for name in limits}
    d_error = 0.
    for i, (r, line, journal) in enumerate(zip(data, raw_lines, ledger), 2):
        assert journal["csv_row"] == line
        assert journal["row_sha256"] == hashlib.sha256(line.encode()).hexdigest()
        assert int(journal["output_row"]) == i and journal["run_id"] == r["run_id"]
        assert journal["source_kind"] == r["source_kind"] == "computed"
        assert all(journal[x] == r[x] == "" for x in ("source_path", "source_row", "source_sha256"))
        p, L, t, b = key(r)
        spec = provenance["patterns"][p]
        assert int(r["m"]) == spec["m"] and r["occupied_bits"] == spec["occupied_bits"]
        assert int(r["class_id"]) == spec["occupancy_class"] and spec["ground_reachable"]
        assert L % (3 * int(r["m"])) == 0 and b == 0 and t >= 0
        assert r["ground_reachable"] == r["premeasurement_ground"] == "true"
        assert r["limit_state"] == "ground_state" and r["projection_method"] == "exact_spectral_support"
        assert float(r["rank_tol"]) == 1e-10
        request = by_job_hash[r["job_file_sha256"]]
        assert key(r) in request["job_keys"]
        m = manifests[r["run_id"]]
        assert m["job_file_sha256"] == request["job_hash"]
        assert r["code_sha256"] == m["code_sha256"]["beta0_study_experiments.jl"]
        g = m["ground_by_key"][(p, L)]
        assert g["ground_reachable"] and g["limit_state"] == "ground_state"
        assert g["occupied_bits"] == r["occupied_bits"]
        for name, bound in limits.items():
            value = abs(float(r[name]))
            assert math.isfinite(value) and value < bound, (p, L, t, name, value)
            maxima[name] = max(maxima[name], value)
        a, c, z, d = (float(r[name]) for name in ("S_A", "S_C", "S_B", "D"))
        assert all(math.isfinite(x) for x in (a, c, z, d)) and min(a, c, z) >= -1e-9
        err = abs((2*a+c)/3-z-d)
        assert err < 1e-10
        d_error = max(d_error, err)
        if math.isinf(t):
            assert abs(float(r["excess_energy"])) < 1e-8 and abs(float(r["residual"])) < 1e-9

    reference_path = ROOT / "data/beta0_study_validation.toml"
    bind(reference_path)
    reference = tomllib.loads(reference_path.read_text(encoding="utf-8"))
    assert reference["status"] == "PASS" and reference["states"] == 99
    bind(reference_path.with_suffix(".csv"), reference["csv_sha256"])
    for name, digest in reference["code_sha256"].items():
        bind(ROOT / name, digest)
    for field in ("max_covariance_error", "max_entropy_error", "max_D_error"):
        assert reference[field] < 1e-7
    for field in ("max_complement_covariance_error", "max_complement_entropy_error", "max_complement_D_error"):
        assert reference[field] < 1e-9
    # Preserve the pilot's data proof before the same live output is appended
    # by the separately frozen confirmation run. Immutable copies, rather than
    # the appendable files, are the pilot receipt's hash-bound data artifacts.
    snapshots = {}
    for source, digest in ((output, output_hash), (ledger_path, ledger_hash)):
        destination = ROOT / "data/provenance/beta0_study" / args.stage / digest / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(source, destination)
        bind(destination, digest)
        snapshots[source.name] = destination.relative_to(ROOT).as_posix()
        if args.stage == "pilot":
            files.pop(source.relative_to(ROOT).as_posix())
    if args.stage == "final":
        prior_path = ANALYSIS / "pilot_data_validation.json"
        bind(prior_path)
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        assert prior["status"] == "PASS" and prior["stage"] == "pilot"
        for name, digest in prior["files"].items():
            bind(ROOT / name, digest)
        pilot_manifests = [m for m in manifests.values() if by_job_hash[m["job_file_sha256"]]["prefix"] == "pilot"]
        assert max(pilot_manifests, key=lambda m: m["finished_utc"])["output_sha256"] == prior["observations_sha256"]
    bind(__file__)
    result = dict(status="PASS", stage=args.stage, generated_utc=datetime.now(timezone.utc).isoformat(),
                  scope="Frozen beta-zero observations and provenance only; no scaling-fit certification",
                  new_rows=len(data), reused_rows=len(reused_keys), target_rows=len(expected),
                  patterns=len({k[0] for k in expected}),
                  observations_sha256=output_hash, ledger_sha256=ledger_hash, snapshots=snapshots,
                  maxima=maxima, max_D_recalculation_error=d_error,
                  independent_reference={k: v for k, v in reference.items() if k.startswith("max_") or k in ("states", "patterns", "scope")},
                  files=files)
    path = ANALYSIS / f"{args.stage}_data_validation.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(f"PASS: {len(data)} new + {len(reused)} reused = {len(expected)} frozen keys; {len(files)} bound files")


if __name__ == "__main__":
    main()
