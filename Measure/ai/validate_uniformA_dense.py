"""Independent validation of dense uniform-A observations and run provenance."""
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
OUT = ROOT / "analysis_uniformA_dense"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    raw = path.read_bytes()
    assert raw.endswith(b"\n"), f"Incomplete CSV: {path}"
    return raw, list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))


class Snapshot:
    """Capture each file once; distinguish concurrent appends from validation errors."""
    def __init__(self):
        self.bytes = {}
        self.output_files = sorted(DATA.glob("uniformA_dense*.csv"))
        self.manifest_files = sorted(DATA.glob("uniformA_dense_run_*.toml"))

    def get(self, path):
        path = Path(path).resolve()
        if path not in self.bytes:
            self.bytes[path] = path.read_bytes()
        return self.bytes[path]

    def csv(self, path, delimiter=","):
        raw = self.get(path)
        assert raw.endswith(b"\n"), f"Incomplete table tail: {path}"
        return raw, list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig")), delimiter=delimiter))

    def changed(self):
        changed = [str(path) for path, content in self.bytes.items()
                   if not path.exists() or path.read_bytes() != content]
        if sorted(DATA.glob("uniformA_dense*.csv")) != self.output_files:
            changed.append("uniformA_dense CSV inventory")
        if sorted(DATA.glob("uniformA_dense_run_*.toml")) != self.manifest_files:
            changed.append("uniformA_dense run manifest inventory")
        return changed


def key(r):
    return int(r["L"]), float(r["theta"]), float(r["beta"])


def validate(stage, snapshot):
    outputs, cache, source_receipts, maxima = {}, {}, {}, {}
    output_run_ids = set()
    output_run_targets = {}
    ledger_cache = {}
    files = snapshot.output_files
    assert files, "No uniform-A observations"
    for path in files:
        path = path.resolve()
        raw, rows = snapshot.csv(path)
        if not rows or "source_kind" not in rows[0]:
            continue
        cache[path.resolve()] = (raw, rows)
        assert len({key(r) for r in rows}) == len(rows), path
        ledger = path.with_name(path.stem + "_provenance.tsv")
        ledger_raw, entries = snapshot.csv(ledger, delimiter="\t")
        ledger_cache[path] = (ledger_raw, entries)
        lines = raw.decode("utf-8-sig").splitlines()
        assert len(entries) == len(rows), (path, "CSV/ledger snapshot row counts differ", len(rows), len(entries))
        reused = computed = 0
        for i, (r, entry) in enumerate(zip(rows, entries), 2):
            output_run_ids.add(r["run_id"])
            output_run_targets.setdefault(r["run_id"], set()).add(path)
            assert int(entry["output_row"]) == i
            assert entry["csv_row"] == lines[i-1]
            assert sha(lines[i-1].encode("utf-8")) == entry["row_sha256"]
            for field in ("source_kind", "source_path", "source_row", "source_sha256", "run_id"):
                assert entry[field] == r[field]
            L, theta, beta = key(r)
            assert L > 0 and L % 3 == 0 and theta >= 0 and math.isfinite(beta) and beta >= 0
            assert r["pattern"] == "uniform_A" and r["occupied_bits"] == "10"
            assert r["premeasurement_ground"].lower() == "true"
            sa, sc, sb, d = (float(r[k]) for k in ("S_A", "S_C", "S_B", "D"))
            assert all(math.isfinite(v) and v >= -1e-8 for v in (sa, sc, sb))
            checks = {"D_identity": abs(d - ((2*sa+sc)/3-sb)),
                      "delta_identity": abs(float(r["delta"]) - (sa-sb)),
                      "orientation_difference": abs(sa-sc)}
            for field in ("premeasurement_excess", "premeasurement_residual", "projector_error",
                          "reconstruction_error", "rectangle_imag_error", "rectangle_real_error_bound"):
                if r.get(field):
                    checks[field] = abs(float(r[field]))
            for field, value in checks.items():
                maxima[field] = max(maxima.get(field, 0.), value)
                assert value < (1e-8 if field in ("premeasurement_excess", "orientation_difference") else 1e-9), (path, i, field, value)
            if r["source_kind"] == "reused":
                reused += 1
                source = Path(r["source_path"]).resolve()
                if source not in cache:
                    cache[source] = snapshot.csv(source)
                sraw, srows = cache[source]
                assert sha(sraw) == r["source_sha256"]
                index = int(r["source_row"])-2
                assert 0 <= index < len(srows), (path, i, "source row out of range")
                sr = srows[index]
                assert key(sr) == key(r) and sr["pattern"] == r["pattern"]
                assert all(float(sr[f]) == float(r[f]) for f in ("S_A", "S_C", "S_B", "D"))
                source_receipts[str(source.relative_to(ROOT))] = sha(sraw)
            else:
                assert r["source_kind"] == "computed"
                computed += 1
        outputs[path.name] = dict(rows=len(rows), computed=computed, reused=reused,
                                  sha256=sha(raw), ledger_sha256=sha(ledger_raw))

    runs = []
    manifest_run_ids = set()
    for path in snapshot.manifest_files:
        manifest_raw = snapshot.get(path)
        record = tomllib.loads(manifest_raw.decode("utf-8"))
        assert record["mode"] == "run"
        assert record["run_id"] not in manifest_run_ids, (path, "duplicate run ID")
        manifest_run_ids.add(record["run_id"])
        requested = set(itertools.product(record["L"], map(float, record["theta"]), record["beta"]))
        if stage == "final":
            assert record["status"] == "complete", path
        target = Path(record["output"]).resolve()
        assert target in ledger_cache, (path, "manifest target is not a validated output CSV", target)
        assert output_run_targets.get(record["run_id"], set()) <= {target}, (path, "run rows occur in a different output CSV")
        expected_ledger = target.with_name(target.stem + "_provenance.tsv")
        assert Path(record["provenance"]).resolve() == expected_ledger, (path, "manifest ledger target mismatch")
        raw, rows = cache[target]
        ledger_raw, entries = ledger_cache[target]
        start = record["already_in_output"]
        count = start + record["new_output_rows"]
        assert 0 <= start <= len(rows) and record["new_output_rows"] >= 0, (path, "invalid output interval")
        if record["status"] == "complete":
            assert count <= len(rows), (path, "completed run output interval is incomplete")
            prefix = b"".join(raw.splitlines(keepends=True)[:count+1])
            assert sha(prefix) == record["output_sha256_at_completion"], path
            ledger_prefix = b"".join(ledger_raw.splitlines(keepends=True)[:count+1])
            assert sha(ledger_prefix) == record["provenance_sha256_at_completion"], (path, "ledger completion prefix hash")
            keys = {key(r) for r in rows[:count]}
            assert requested <= keys, (path, requested-keys)
        code_sources = {}
        for name, digest in record["code_sha256"].items():
            candidates = [ROOT/name, DATA/"provenance"/"code"/digest/name]
            match = next((p for p in candidates if p.exists() and sha(snapshot.get(p)) == digest), None)
            assert match is not None, (path, name, digest)
            code_sources[name] = str(match.relative_to(ROOT))
        indices = [i for i, r in enumerate(rows) if r["run_id"] == record["run_id"]]
        assert indices == list(range(start, min(count, len(rows)))), (path, "run rows do not occupy their declared contiguous interval")
        run_rows = [rows[i] for i in indices]
        assert all(key(r) in requested for r in run_rows), (path, "row outside run grid")
        assert all(r["code_sha256"] == record["code_sha256"]["uniformA_dense_experiments.jl"] for r in run_rows)
        if record["status"] == "complete":
            assert len(run_rows) == record["new_output_rows"]
        run_computed = sum(r["source_kind"] == "computed" for r in run_rows)
        run_reused = sum(r["source_kind"] == "reused" for r in run_rows)
        assert record["new_compute_rows"] + record["reuse_rows"] == record["new_output_rows"], (path, "manifest source counts")
        if record["status"] == "complete":
            assert (run_computed, run_reused) == (record["new_compute_rows"], record["reuse_rows"]), (path, "computed/reused run counts")
        else:
            assert run_computed <= record["new_compute_rows"] and run_reused <= record["reuse_rows"], (path, "running source counts exceed plan")
        declared_sources = {}
        for source in record.get("sources", []):
            source_path = Path(source["path"]).resolve()
            if source_path not in cache:
                cache[source_path] = snapshot.csv(source_path)
            source_raw, source_rows = cache[source_path]
            assert sha(source_raw) == source["sha256"] and len(source_rows) == source["rows"], (path, "declared input changed", source_path)
            declared_sources[source_path] = source["sha256"]
        method = record.get("entropy_method")
        for row in run_rows:
            if row["source_kind"] == "reused":
                source_path = Path(row["source_path"]).resolve()
                assert declared_sources.get(source_path) == row["source_sha256"], (path, "reuse source missing from manifest")
                if method is not None:
                    assert row.get("entropy_method") == "source_recorded", (path, "reuse method label")
            elif method is not None:
                assert row.get("entropy_method") == method, (path, "computed method label")
        half_check_count = 0
        # Running manifests are written before the scan and finalized only at its end.
        # Their in-memory structural checks are not yet persisted and cannot be audited.
        if method == "halfcomplex" and record["status"] == "complete":
            assert "uniformA_entropy_extension.jl" in record["code_sha256"], (path, "missing half extension provenance")
            checks = record.get("entropy_checks", [])
            checks_by_row = {int(check["output_row"]): check for check in checks}
            expected_rows = {i+2 for i in indices if rows[i]["source_kind"] == "computed"}
            assert len(checks_by_row) == len(checks) and set(checks_by_row) == expected_rows, (path, "half-complex check coverage")
            for output_row, check in checks_by_row.items():
                row = rows[output_row-2]
                assert key(check) == key(row), (path, output_row, "half check parameter mismatch")
                for field in ("max_structure_error", "orbital_projector_error"):
                    value = float(check[field])
                    assert math.isfinite(value) and 0 <= value <= 1e-10, (path, output_row, field, value)
                    maxima["half_"+field] = max(maxima.get("half_"+field, 0.), value)
                half_dimension = 2 * int(row["L"])**2 // 9
                assert check["workspace_matrix_bytes"] == 32 * half_dimension**2, (path, output_row, "half matrix workspace size")
            half_check_count = len(checks)
        runs.append(dict(manifest=path.name, status=record["status"], rows=len(run_rows),
                         computed=run_computed, reused=run_reused,
                         half_structure_checks=half_check_count,
                         half_checks_pending=method == "halfcomplex" and record["status"] != "complete",
                         historical_method_field_absent=method is None,
                         manifest_sha256=sha(manifest_raw), code_sources=code_sources))

    assert output_run_ids <= manifest_run_ids, ("observations without run manifest", output_run_ids-manifest_run_ids)

    changed = snapshot.changed()
    assert stage != "final" or not changed, ("Sources changed during final validation", changed)
    return dict(status="PASS", stage=stage, generated_utc=datetime.now(timezone.utc).isoformat(),
                   outputs=outputs, runs=runs, source_sha256=source_receipts, numerical_maxima=maxima,
                   files_changed_during_validation=changed,
                   snapshot_scope="Captured bytes; preliminary PASS does not imply scans are complete. Running half-complex manifests have structural checks pending.",
                   validator_sha256=sha(Path(__file__).read_bytes()),
                   scope="Raw observations, reuse and run provenance; no assertion of asymptotic scaling.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["preliminary", "final"], default="final")
    args = parser.parse_args()
    retries = []
    for attempt in range(3):
        snapshot = Snapshot()
        try:
            receipt = validate(args.stage, snapshot)
            break
        except (AssertionError, ValueError, KeyError, IndexError, OSError) as error:
            changed = snapshot.changed()
            if args.stage == "preliminary" and changed and attempt < 2:
                retries.append(dict(error=str(error), changed_files=changed))
                continue
            receipt = dict(status="SNAPSHOT_RACE" if changed else "FAIL", stage=args.stage,
                           generated_utc=datetime.now(timezone.utc).isoformat(),
                           error=str(error), files_changed_during_validation=changed,
                           validator_sha256=sha(Path(__file__).read_bytes()),
                           scope="No PASS issued. Changed inputs indicate snapshot competition; otherwise the captured validation is inconsistent.")
            break
    receipt["snapshot_retries"] = retries
    OUT.mkdir(exist_ok=True)
    (OUT/"independent_data_validation.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    if receipt["status"] == "PASS":
        print(f"PASS: {sum(v['rows'] for v in receipt['outputs'].values())} recorded rows, {len(receipt['runs'])} runs, {args.stage}; {len(retries)} snapshot retries")
    else:
        print(json.dumps(receipt, indent=2))
        raise SystemExit(2 if receipt["status"] == "SNAPSHOT_RACE" else 1)


if __name__ == "__main__":
    main()
