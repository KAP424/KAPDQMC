"""Audit additional-pattern observations, exact reuse, run receipts and planned grids.

This independently checks saved records; it does not assert asymptotic scaling.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import tomllib

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "analysis_beta_patterns"
CATALOGUE = {
    "one_B_2": (2, "special", "10101001"),
    # Early small-system runner validation used this control before the main catalogue was locked.
    "one_B_3": (3, "special", "101010101010101001"),
    "diagonal_B_3": (3, "special", "011010100110101001"),
    "random_half_3": (3, "general", "100110001011010110"),
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def key(r):
    return r["pattern"], int(r["L"]), float(r["theta"]), float(r["beta"])


class Snapshot:
    def __init__(self):
        self.bytes = {}
        self.outputs = sorted(DATA.glob("beta_patterns*.csv"))
        self.manifests = sorted(DATA.glob("beta_patterns_run_*.toml"))

    def get(self, path):
        path = Path(path).resolve()
        if path not in self.bytes:
            self.bytes[path] = path.read_bytes()
        return self.bytes[path]

    def table(self, path, delimiter=","):
        raw = self.get(path)
        assert raw.endswith(b"\n"), (path, "incomplete tail")
        return raw, list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig")), delimiter=delimiter))

    def changed(self):
        changed = [str(p) for p, b in self.bytes.items() if not p.exists() or p.read_bytes() != b]
        if sorted(DATA.glob("beta_patterns*.csv")) != self.outputs:
            changed.append("CSV inventory")
        if sorted(DATA.glob("beta_patterns_run_*.toml")) != self.manifests:
            changed.append("run inventory")
        return changed


def validate(stage, snap):
    assert snap.outputs, "No additional-pattern observations"
    outputs, cache, ledgers, maxima, run_targets = {}, {}, {}, {}, {}
    for path in snap.outputs:
        path = path.resolve()
        raw, rows = snap.table(path)
        if not rows or "source_kind" not in rows[0]:
            continue
        cache[path] = raw, rows
        ledger = path.with_name(path.stem + "_provenance.tsv")
        lraw, entries = snap.table(ledger, "\t")
        ledgers[path] = lraw, entries
        lines = raw.decode("utf-8-sig").splitlines()
        assert len(entries) == len(rows)
        assert len({key(r) for r in rows}) == len(rows), (path, "duplicate key")
        for i, (r, e) in enumerate(zip(rows, entries), 2):
            assert int(e["output_row"]) == i and e["csv_row"] == lines[i-1]
            assert e["row_sha256"] == sha(lines[i-1].encode())
            assert all(e[f] == r[f] for f in ("run_id", "source_kind", "source_path", "source_row", "source_sha256"))
            run_targets.setdefault(r["run_id"], set()).add(path)
            pattern, L, theta, beta = key(r)
            m, family, bits = CATALOGUE[pattern]
            assert (int(r["m"]), r["family"], r["occupied_bits"]) == (m, family, bits)
            assert L >= 3 and L % (3*m) == 0 and theta >= 0 and math.isfinite(beta) and beta >= 0
            assert r["premeasurement_ground"].lower() == "true" and float(r["rank_tol"]) == 1e-10
            sa, sc, sb, d = (float(r[f]) for f in ("S_A", "S_C", "S_B", "D"))
            assert all(math.isfinite(v) and v >= -1e-8 for v in (sa, sc, sb))
            checks = {"D_identity": abs(d-((2*sa+sc)/3-sb)), "delta_identity": abs(float(r["delta"])-(sa-sb))}
            for f in ("premeasurement_excess", "premeasurement_residual", "projector_error", "reconstruction_error", "rectangle_imag_error", "rectangle_real_error_bound"):
                if r[f]:
                    checks[f] = abs(float(r[f]))
            for f, v in checks.items():
                assert math.isfinite(v) and v < (1e-8 if f == "premeasurement_excess" else 1e-9), (path, i, f, v)
                maxima[f] = max(maxima.get(f, 0), v)
            # SA=SC is NOT assumed for general supercells.
            if r["source_kind"] == "reused":
                source = Path(r["source_path"]).resolve()
                if source not in cache:
                    cache[source] = snap.table(source)
                sraw, srows = cache[source]
                assert sha(sraw) == r["source_sha256"]
                sr = srows[int(r["source_row"])-2]
                assert key(sr) == key(r)
                assert all(sr[f] == r[f] for f in ("S_A", "S_C", "S_B", "D")), (path, i, "reuse changed values")
                assert r["entropy_method"] == "source_recorded"
            else:
                assert r["source_kind"] == "computed" and r["entropy_method"] == "real_cholesky"
        outputs[path.name] = dict(rows=len(rows), computed=sum(r["source_kind"] == "computed" for r in rows),
                                  reused=sum(r["source_kind"] == "reused" for r in rows), sha256=sha(raw), ledger_sha256=sha(lraw))
    runs, run_ids, ground_checks, folded_checks = [], set(), [], 0
    for path in snap.manifests:
        mraw = snap.get(path)
        rec = tomllib.loads(mraw.decode())
        rid = rec["run_id"]
        assert rid not in run_ids and rec["mode"] == "run"
        run_ids.add(rid)
        if stage == "final":
            assert rec["status"] in ("complete", "interrupted"), path
        target = Path(rec["output"]).resolve()
        assert target in ledgers and run_targets.get(rid, set()) <= {target}
        raw, rows = cache[target]
        lraw, entries = ledgers[target]
        assert Path(rec["provenance"]).resolve() == target.with_name(target.stem+"_provenance.tsv")
        requested = set(itertools.product(rec["patterns"], rec["L"], map(float, rec["theta"]), rec["beta"]))
        start, count = rec["already_in_output"], rec["new_output_rows"]
        end = start+count
        assert 0 <= start <= len(rows) and count >= 0
        interrupted = rec["status"] == "interrupted"
        receipt = None
        if interrupted:
            receipt_raw = snap.get(rec["interruption_receipt"])
            assert sha(receipt_raw) == rec["interruption_receipt_sha256"]
            receipt = json.loads(receipt_raw)
            assert receipt["status"] == "interrupted_at_verified_complete_csv_ledger_prefix"
            assert receipt["run_id"] == rid and Path(receipt["manifest"]).resolve() == path.resolve()
            originals = {}
            for saved in receipt["snapshots"]:
                content = snap.get(saved["snapshot_path"])
                assert sha(content) == saved["sha256"] and len(content) == saved["bytes"]
                originals[Path(saved["source_path"]).resolve()] = content
            original_manifest = originals[path.resolve()]
            assert sha(original_manifest) == receipt["original_manifest_sha256"]
            original = tomllib.loads(original_manifest.decode())
            assert original["status"] == "running"
            assert {k: v for k, v in rec.items() if k not in ("status", "interruption_receipt", "interruption_receipt_sha256")} == {k: v for k, v in original.items() if k != "status"}
            assert receipt["code_sha256"] == rec["code_sha256"]
            recorded_end = receipt["output_rows_at_interruption"]
            assert start <= recorded_end <= min(end, len(rows))
            assert recorded_end-start == receipt["completed_rows_this_run"]
            prefix = b"".join(raw.splitlines(keepends=True)[:recorded_end+1])
            ledger_prefix = b"".join(lraw.splitlines(keepends=True)[:recorded_end+1])
            assert prefix == originals[target] and sha(prefix) == receipt["output_sha256_at_interruption"]
            assert ledger_prefix == originals[target.with_name(target.stem+"_provenance.tsv")]
            assert sha(ledger_prefix) == receipt["provenance_sha256_at_interruption"]
            assert receipt["completed_output_rows"] == list(range(start+2, recorded_end+2))
            assert key(receipt["last_output_key"]) == key(rows[recorded_end-1])
        else:
            recorded_end = min(end, len(rows))
        indices = [i for i, r in enumerate(rows) if r["run_id"] == rid]
        assert indices == list(range(start, recorded_end))
        run_rows = [rows[i] for i in indices]
        assert all(key(r) in requested for r in run_rows)
        assert all(r["code_sha256"] == rec["code_sha256"]["beta_patterns_experiments.jl"] for r in run_rows)
        for name, digest in rec["code_sha256"].items():
            source = DATA/"provenance"/"code"/digest/name
            assert sha(snap.get(source)) == digest, (path, name, "code snapshot")
        declared = {}
        for source in rec["sources"]:
            sp = Path(source["path"]).resolve()
            sraw, srows = snap.table(sp)
            assert sha(sraw) == source["sha256"] and len(srows) == source["rows"]
            declared[sp] = source["sha256"]
        for r in run_rows:
            if r["source_kind"] == "reused":
                assert declared[Path(r["source_path"]).resolve()] == r["source_sha256"]
        assert rec["new_compute_rows"]+rec["reuse_rows"] == count
        if rec["status"] == "complete":
            assert end <= len(rows) and len(run_rows) == count
            assert sha(b"".join(raw.splitlines(keepends=True)[:end+1])) == rec["output_sha256_at_completion"]
            assert sha(b"".join(lraw.splitlines(keepends=True)[:end+1])) == rec["provenance_sha256_at_completion"]
            assert requested <= {key(r) for r in rows[:end]}
            assert sum(r["source_kind"] == "computed" for r in run_rows) == rec["new_compute_rows"]
            checked = {(r["pattern"], int(r["L"])) for r in rec["ground_checks"]}
            assert checked == {(r["pattern"], int(r["L"])) for r in run_rows}
            for g in rec["ground_checks"]:
                assert abs(g["excess_energy"]) < 1e-8 and g["residual"] < 1e-9 and abs(g["accessible_excess"]) < 1e-8
                ground_checks.append(dict(run_id=rid, **g))
            checks = {int(r["output_row"]): r for r in rec["entropy_checks"]}
            expected = {i+2 for i in indices if rows[i]["source_kind"] == "computed"}
            assert len(checks) == len(rec["entropy_checks"]) and set(checks) == expected
            for index, check in checks.items():
                assert key(check) == key(rows[index-2])
                assert 0 <= check["max_imaginary"] <= 2e-10
                assert 0 <= check["real_part_entropy_bound"] <= 1e-9
                assert 0 <= check["orbital_projector_error"] <= 1e-9
            if rec.get("strip_method") == "folded":
                assert "beta_patterns_strip_extension.jl" in rec["code_sha256"]
                assert rec["strip_threads"] >= 1 and rec["rectangle_threads"] == rec["blas_threads"]
                strip_checks = {int(r["output_row"]): r for r in rec["strip_checks"]}
                assert len(strip_checks) == len(rec["strip_checks"]) and set(strip_checks) == expected
                for index, check in strip_checks.items():
                    row = rows[index-2]
                    assert key(check) == key(row)
                    m, L = int(row["m"]), int(row["L"])
                    q = L//m
                    for orientation in ("orientation_A", "orientation_C"):
                        detail = check[orientation]
                        assert detail["method"] == "conjugate_momentum_fold"
                        assert detail["q"] == detail["original_entropy_evaluations"] == detail["total_weight"] == q
                        assert detail["block_dimension"] == 2*m*(L//3)
                        assert detail["entropy_evaluations"] == q//2+1
                        assert detail["trim_count"] == 1+int(q % 2 == 0)
                        assert detail["probe_parallel"] == (1 if q > 2 else 0)
                        assert detail["probe_partner"] == -detail["probe_parallel"] % q
                        assert 0 < detail["conjugacy_atol"] <= 2e-10 and 0 < detail["block_atol"] <= 2e-10
                        for field, tolerance in (("max_bloch_conjugacy", detail["conjugacy_atol"]),
                                                 ("max_probe_conjugacy", detail["block_atol"]),
                                                 ("max_trim_imaginary", detail["block_atol"])):
                            value = detail[field]
                            assert math.isfinite(value) and 0 <= value <= tolerance
                            maxima[field] = max(maxima.get(field, 0), value)
                    folded_checks += 1
            else:
                assert rec.get("strip_method", "original") == "original"
        runs.append(dict(manifest=path.name, manifest_sha256=sha(mraw), status=rec["status"], rows=len(run_rows),
                         interruption_receipt_sha256=rec.get("interruption_receipt_sha256"),
                         interruption_scope="Saved row checks and verified checkpoint; in-memory manifest checks were not persisted" if interrupted else None))
    assert set(run_targets) <= run_ids, "Rows without run manifests"
    folding_comparison = None
    comparison_path = DATA/"beta_patterns_folded_validation_comparison.json"
    if comparison_path.exists():
        comparison_raw = snap.get(comparison_path)
        comparison = json.loads(comparison_raw)
        reference_raw = snap.get(comparison["reference_path"])
        test_raw, test_rows = snap.table(comparison["observations_path"])
        assert sha(reference_raw) == comparison["reference_sha256"]
        assert sha(test_raw) == comparison["observations_sha256"]
        reference = tomllib.loads(reference_raw.decode())
        reference_rows = {key(r): r for r in reference["observations"]}
        assert reference["status"] == "complete" and comparison["status"] == "PASS"
        assert len(reference_rows) == len(test_rows) == comparison["rows"] == 18
        assert set(reference_rows) == {key(r) for r in test_rows}
        maximum_error = max(abs(float(row[field])-float(reference_rows[key(row)][field]))
                            for row in test_rows for field in ("S_A", "S_C", "S_B", "D"))
        assert maximum_error == comparison["max_S_A_S_C_S_B_D_error"]
        assert maximum_error <= comparison["tolerance"] <= 1e-9
        assert max(abs(r["S_B"]-r["eigen_reference"]) for r in reference_rows.values()) <= 1e-9
        folding_comparison = dict(rows=18, max_entropy_error=maximum_error, receipt_sha256=sha(comparison_raw))
    if folded_checks:
        assert folding_comparison is not None
    planraw = snap.get(OUT/"plan.json")
    plan = json.loads(planraw)
    required = set()
    for grid in plan["required_finite_grids"]+plan["required_equilibrium_grids"]:
        required.update(itertools.product(grid["patterns"], grid["L"], map(float, grid["theta"]), grid["beta"]))
    primary_path = (DATA/"beta_patterns_observations.csv").resolve()
    available = {key(r) for r in cache.get(primary_path, (None, []))[1]}
    missing = required-available
    if stage == "final":
        assert not missing, ("missing required grid", len(missing))
        assert not snap.changed(), "Files changed during final audit"
    return dict(status="PASS", stage=stage, generated_utc=datetime.now(timezone.utc).isoformat(),
                outputs=outputs, runs=runs, numerical_maxima=maxima, ground_checks=ground_checks,
                folded_strip_rows_checked=folded_checks,
                independent_folding_comparison=folding_comparison,
                required_rows=len(required), missing_required_rows=len(missing), plan_sha256=sha(planraw),
                validator_sha256=sha(Path(__file__).read_bytes()),
                scope="Saved observations, run/source/code receipts, GS-limit checks, and requested-grid coverage. No assumption SA=SC; no claim of asymptotic scaling.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["preliminary", "final"], default="final")
    args = parser.parse_args()
    for attempt in range(3):
        snap = Snapshot()
        try:
            receipt = validate(args.stage, snap)
            break
        except (AssertionError, KeyError, ValueError, IndexError, OSError) as error:
            changed = snap.changed()
            if changed and args.stage == "preliminary" and attempt < 2:
                continue
            receipt = dict(status="SNAPSHOT_RACE" if changed else "FAIL", error=str(error), changed=changed)
            break
    OUT.mkdir(exist_ok=True)
    (OUT/"independent_data_validation.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ("status", "stage", "required_rows", "missing_required_rows", "error") if k in receipt}))
    if receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
