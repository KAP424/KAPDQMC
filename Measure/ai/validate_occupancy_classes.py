"""Independent saved-data audit for the three occupation classes.

Preliminary reads only complete CSV/ledger prefixes and reports missing work.
Final requires the complete plan and every read file to remain unchanged.
Neither mode starts simulations, changes inputs, or validates scaling claims.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict
import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import re
import tomllib

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "analysis_occupancy_classes"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def key(row):
    return row["pattern"], int(row["L"]), float(row["theta"]), round(float(row["beta"]), 10)


def json_key(k):
    return [k[0], k[1], "Inf" if math.isinf(k[2]) else k[2], k[3]]


class Snapshot:
    def __init__(self, stage):
        self.stage = stage
        self.bytes = {}
        self.truncated = {}
        self.manifest_inventory = sorted(DATA.glob("occupancy_classes_run_*.toml"))

    def get(self, path):
        path = Path(path).resolve()
        if path not in self.bytes:
            self.bytes[path] = path.read_bytes()
        return self.bytes[path]

    def table(self, path, delimiter=","):
        path = Path(path).resolve()
        raw = self.get(path)
        end = raw.rfind(b"\n") + 1
        if end != len(raw):
            assert self.stage == "preliminary", (path.name, "incomplete final line")
            self.truncated[str(path)] = len(raw) - end
        prefix = raw[:end]
        assert prefix, (path.name, "no complete header")
        lines = prefix.decode("utf-8-sig").splitlines()
        rows = list(csv.DictReader(io.StringIO("\n".join(lines)), delimiter=delimiter))
        assert all(None not in r and None not in r.values() for r in rows), (path, "malformed complete row")
        return prefix, rows, lines

    def changed(self):
        changed = [str(p) for p, raw in self.bytes.items() if not p.exists() or p.read_bytes() != raw]
        if sorted(DATA.glob("occupancy_classes_run_*.toml")) != self.manifest_inventory:
            changed.append("occupancy_classes run-manifest inventory")
        return changed


def validate(stage, snap):
    planraw = snap.get(OUT / "plan.json")
    plan = json.loads(planraw)
    metadata = plan["pattern_metadata"]
    assert len(plan["patterns"]) == len(set(plan["patterns"])) == 6
    assert set(metadata) == set(plan["patterns"])
    assert Counter(m["occupancy_class"] for m in metadata.values()) == {1: 2, 2: 2, 3: 2}
    catalogue = snap.get(ROOT / "occupancy_classes_patterns.jl").decode()
    declared = {
        name: (family, int(m), bits, int(cls))
        for name, family, m, bits, cls in re.findall(
            r'\("([^"]+)",\s*"([^"]+)",\s*(\d+),\s*"([01]+)",\s*(\d+)\)', catalogue)
    }
    classes = []
    for name, meta in metadata.items():
        bits, m = meta["occupied_bits"], meta["m"]
        assert declared[name] == (meta["family"], m, bits, meta["occupancy_class"])
        assert len(bits) == 2*m*m and bits.count("1") == m*m and set(bits) <= {"0", "1"}
        cells = [int(bits[i])+int(bits[i+1]) for i in range(0, len(bits), 2)]
        counts = Counter(cells)
        actual_class = 1 if counts[1] == m*m else 2 if counts[1] == 0 else 3
        assert actual_class == meta["occupancy_class"] and counts[0] == counts[2]
        classes.append(dict(pattern=name, occupancy_class=actual_class,
                            n0=counts[0], n1=counts[1], n2=counts[2]))

    physicsraw = snap.get(DATA / "occupancy_classes_validation.toml")
    physics = tomllib.loads(physicsraw.decode())
    assert physics["status"] == "PASS"
    assert sha(snap.get(DATA / "occupancy_classes_validation.csv")) == physics["csv_sha256"]
    for name, digest in physics["code_sha256"].items():
        assert sha(snap.get(ROOT / name)) == digest, (name, "changed since small-system validation")
    fe_hash = physics["code_sha256"]["fermion_experiment.jl"]
    fe_code = snap.get(ROOT / "fermion_experiment.jl").decode()
    assert "F, E = findall(occ), findall(.!occ)" in fe_code
    assert "raw[pre.E,:] .*= exp(-2beta)" in fe_code
    assert "exp.(-theta .* prepared.gaps[k])" in fe_code

    cache, inputs, maxima = {}, {}, {}
    all_rows, audit_rows, source_counts, missing_fields = [], {}, Counter(), Counter()
    duplicates = []
    by_key = {}

    def table(path):
        path = Path(path).resolve()
        if path not in cache:
            cache[path] = snap.table(path, delimiter="\t" if path.suffix == ".tsv" else ",")
        return cache[path]

    def check_value(field, value, limit, identity):
        value = abs(float(value))
        assert math.isfinite(value) and value < limit, (identity, field, value, limit)
        maxima[field] = max(maxima.get(field, 0.), value)

    def inspect_row(row, identity, strict):
        name, L, theta, beta = key(row)
        meta = metadata[name]
        assert (int(row["m"]), row["family"]) == (meta["m"], meta["family"]), identity
        if row.get("occupied_bits"):
            assert row["occupied_bits"] == meta["occupied_bits"], (identity, "bits")
        else:
            missing_fields["occupied_bits"] += 1
            assert not strict, (identity, "missing bits")
        assert L >= 3 and L % (3*meta["m"]) == 0, (identity, "incompatible size/cut")
        assert theta >= 0 and not math.isnan(theta) and math.isfinite(beta) and beta >= 0
        flag = row.get("premeasurement_ground", row.get("ground", "")).lower()
        assert flag == "true", (identity, "ground-limit flag")
        sa, sc, sb, d = (float(row[f]) for f in ("S_A", "S_C", "S_B", "D"))
        assert all(math.isfinite(x) and x >= -1e-8 for x in (sa, sc, sb)), identity
        check_value("D_identity", d-((2*sa+sc)/3-sb), 1e-9, identity)
        if row.get("delta"):
            check_value("delta_identity", float(row["delta"])-(sa-sb), 1e-9, identity)
        for field, alternate, limit in (
            ("premeasurement_excess", "premeasurement_limit_excess", 1e-8),
            ("premeasurement_residual", "premeasurement_limit_residual", 1e-9),
            ("projector_error", None, 1e-10),
            ("reconstruction_error", None, 1e-9),
            ("rectangle_imag_error", None, 2e-10),
            ("rectangle_real_error_bound", None, 1e-9),
            ("accessible_excess", None, 1e-8),
        ):
            value = row.get(field, row.get(alternate, "") if alternate else "")
            if value:
                check_value(field, value, limit, identity)
            else:
                missing_fields[field] += 1
                # Some reused old sources did not record rectangle diagnostics.
                if strict and field not in ("rectangle_imag_error", "rectangle_real_error_bound"):
                    raise AssertionError((identity, "missing", field))
        if row.get("rank_tol"):
            assert float(row["rank_tol"]) == 1e-10
        if math.isinf(theta) and beta == 0:
            for field, limit in (("excess_energy", 1e-8), ("residual", 1e-9)):
                if row.get(field):
                    check_value("unmeasured_ground_"+field, row[field], limit, identity)
        # The flag certifies the theta->Inf limit, not the finite-theta state.
        if math.isfinite(theta) and beta == 0 and 4 <= theta <= 8 and L >= 144:
            if row.get("residual"):
                assert float(row["residual"]) > 1e-8, (identity, "finite time replaced by ground state")

    for relative in plan["inputs"]:
        path = (ROOT / relative).resolve()
        raw, rows, lines = table(path)
        ledger_path = path.with_name(path.stem+"_provenance.tsv")
        ledger = None
        unpaired = 0
        if ledger_path.exists():
            lraw, ledger, llines = table(ledger_path)
            assert stage == "preliminary" or len(rows) == len(ledger), (path.name, "ledger length")
            paired = min(len(rows), len(ledger))
            unpaired = len(rows) - paired
            rows = rows[:paired]
            for index, (row, entry) in enumerate(zip(rows, ledger[:paired]), 2):
                assert int(entry["output_row"]) == index
                assert entry["csv_row"] == lines[index-1]
                assert entry["row_sha256"] == sha(lines[index-1].encode())
                assert all(entry[f] == row[f] for f in
                           ("run_id", "source_kind", "source_path", "source_row", "source_sha256"))
            assert len({key(row) for row in rows}) == len(rows), (path.name, "duplicate output keys")
            audit_rows[path] = rows
        else:
            lraw = None
        selected = []
        for index, row in enumerate(rows, 2):
            if row.get("pattern") not in metadata:
                continue
            identity = (path.name, index)
            inspect_row(row, identity, strict=ledger is not None)
            selected.append(row)
            if row.get("source_kind") == "reused":
                source = Path(row["source_path"]).resolve()
                sraw, srows, _ = table(source)
                assert sha(sraw) == row["source_sha256"], (identity, "reuse source hash")
                source_index = int(row["source_row"])-2
                assert 0 <= source_index < len(srows)
                sr = srows[source_index]
                assert key(sr) == key(row)
                assert all(sr[f] == row[f] for f in ("S_A", "S_C", "S_B", "D"))
                assert row["entropy_method"] == "source_recorded"
            elif row.get("source_kind"):
                assert row["source_kind"] == "computed" and row["entropy_method"] == "real_cholesky"
            source_counts[row.get("source_kind", "legacy_reference")] += 1
            k = key(row)
            if k in by_key:
                previous, previous_identity = by_key[k]
                error = max(abs(float(row[f])-float(previous[f])) for f in ("S_A", "S_C", "S_B", "D"))
                assert error < 1e-8, (identity, previous_identity, "conflicting duplicate", error)
                duplicates.append(dict(key=json_key(k), first=list(previous_identity), second=list(identity), max_error=error))
                maxima["duplicate_entropy_difference"] = max(maxima.get("duplicate_entropy_difference", 0.), error)
            else:
                by_key[k] = row, identity
            all_rows.append(row)
        inputs[relative] = dict(complete_rows=len(table(path)[1]), accepted_rows=len(rows),
            selected_rows=len(selected), file_sha256=sha(snap.get(path)), complete_prefix_sha256=sha(raw),
            ledger_sha256=sha(lraw) if lraw else None, ledger_verified=ledger is not None,
            unpaired_csv_rows_deferred=unpaired)

    runs, verified_code, finite_rows, inf_rows = [], set(), 0, 0
    for path, rows in audit_rows.items():
        grouped = defaultdict(list)
        for index, row in enumerate(rows, 2):
            grouped[row["run_id"]].append((index, row))
        prefix = "occupancy_classes" if path.name.startswith("occupancy_classes") else "beta_patterns"
        for rid, records in grouped.items():
            manifestpath = DATA / f"{prefix}_run_{rid}.toml"
            mraw = snap.get(manifestpath)
            manifest = tomllib.loads(mraw.decode())
            assert manifest["run_id"] == rid and Path(manifest["output"]).resolve() == path
            assert manifest["status"] in ("running", "complete", "interrupted")
            if stage == "final":
                assert manifest["status"] != "running", (manifestpath.name, "still running")
            hashes = manifest["code_sha256"]
            assert hashes["fermion_experiment.jl"] == fe_hash, (manifestpath.name, "unvalidated FE kernel")
            runner = prefix+"_experiments.jl"
            for name, digest in hashes.items():
                snapshotpath = DATA / "provenance" / "code" / digest / name
                content = snap.get(snapshotpath)
                assert sha(content) == digest, (manifestpath.name, name, "code snapshot")
                verified_code.add((name, digest))
                if name == runner:
                    text = re.sub(r"\s+", "", content.decode())
                    assert "FE.evolve(prepared,theta)" in text and "FE.measure(state,beta)" in text
            if prefix == "occupancy_classes":
                assert hashes["occupancy_classes_patterns.jl"] == sha(snap.get(ROOT/"occupancy_classes_patterns.jl"))
            requested = set(itertools.product(manifest["patterns"], manifest["L"],
                                             map(float, manifest["theta"]), manifest["beta"]))
            for _, row in records:
                assert row["code_sha256"] == hashes[runner]
                assert key(row) in requested
                finite_rows += int(math.isfinite(float(row["theta"])))
                inf_rows += int(math.isinf(float(row["theta"])))
            if manifest["status"] == "complete":
                end = max(i for i, _ in records)
                raw = table(path)[0]
                if prefix == "beta_patterns":
                    end = manifest["total_rows_at_completion"] + 1
                    expected_hash = manifest["output_sha256_at_completion"]
                    assert len(records) == manifest["new_output_rows"]
                    ledgerraw = table(path.with_name(path.stem+"_provenance.tsv"))[0]
                    assert sha(b"".join(ledgerraw.splitlines(keepends=True)[:end])) == manifest["provenance_sha256_at_completion"]
                else:
                    expected_hash = manifest["output_sha256"]
                    assert len(records) == manifest["new_rows"]
                assert sha(b"".join(raw.splitlines(keepends=True)[:end])) == expected_hash
                checks = {(g["pattern"], int(g["L"])): g for g in manifest["ground_checks"]}
                assert {(r["pattern"], int(r["L"])) for _, r in records} <= set(checks)
                for g in checks.values():
                    assert abs(float(g.get("excess", g.get("excess_energy")))) < 1e-8
                    assert abs(float(g["residual"])) < 1e-9
            elif manifest["status"] == "interrupted":
                receipt = snap.get(manifest["interruption_receipt"])
                assert sha(receipt) == manifest["interruption_receipt_sha256"]
                interruption = json.loads(receipt)
                assert interruption["run_id"] == rid
                assert interruption["status"] == "interrupted_at_verified_complete_csv_ledger_prefix"
                assert interruption["completed_rows_this_run"] == len(records)
                for item in interruption["snapshots"]:
                    saved = snap.get(item["snapshot_path"])
                    assert sha(saved) == item["sha256"]
                    original = Path(item["source_path"]).resolve()
                    if original in (path, path.with_name(path.stem+"_provenance.tsv")):
                        end = interruption["output_rows_at_interruption"] + 1
                        assert saved == b"".join(table(original)[0].splitlines(keepends=True)[:end])
            runs.append(dict(manifest=manifestpath.name, sha256=sha(mraw),
                             status=manifest["status"], accepted_rows=len(records)))

    required = set()
    for grid in plan["required_finite_grids"]+plan["required_equilibrium_grids"]:
        required.update(itertools.product(grid["patterns"], grid["L"], map(float, grid["theta"]), grid["beta"]))
    missing = required-set(by_key)
    coverage = []
    for name in plan["patterns"]:
        needed = {k for k in required if k[0] == name}
        coverage.append(dict(pattern=name, required=len(needed), available=len(needed-missing),
                             missing=len(needed & missing)))
    if stage == "final":
        assert not missing, ("incomplete planned grid", len(missing))
        for manifestpath in snap.manifest_inventory:
            manifest = tomllib.loads(snap.get(manifestpath).decode())
            assert manifest["status"] == "complete", (manifestpath.name, "unfinished new run")
        assert not snap.changed(), ("files changed during final audit", snap.changed())
    return dict(
        status="PASS" if stage == "final" else "PASS_PRELIMINARY",
        stage=stage, generated_utc=datetime.now(timezone.utc).isoformat(),
        plan_sha256=sha(planraw), validator_sha256=sha(snap.get(Path(__file__))),
        required_rows=len(required), available_required_rows=len(required-missing),
        missing_required_rows=len(missing),
        missing_keys=[json_key(k) for k in sorted(missing)],
        coverage=coverage, classes=classes, inputs=inputs, source_counts=dict(source_counts),
        audited_runs=runs, code_snapshots_checked=len(verified_code),
        ledger_finite_rows=finite_rows, ledger_infinite_rows=inf_rows,
        numerical_maxima=maxima, consistent_duplicate_rows=len(duplicates),
        duplicates=duplicates, legacy_missing_fields=dict(missing_fields),
        small_system_validation_sha256=sha(physicsraw),
        incomplete_tail_bytes=snap.truncated,
        changed_during_preliminary=snap.changed() if stage == "preliminary" else [],
        sources={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha(raw)
                 for p, raw in snap.bytes.items()},
        scope="Data identity, six occupation classes, own-empty protocol via validated kernel hashes, finite/Inf distinction, CSV/ledger/source/run/code provenance, ground-limit diagnostics and plan coverage. The ground flag describes reachability of theta->Inf, not finite-theta convergence. Legacy reference tables without ledgers are labeled; audited beta/new outputs have full row provenance. No scaling-fit or asymptotic claim."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preliminary", "final"), default="preliminary")
    args = parser.parse_args()
    for attempt in range(3):
        snap = Snapshot(args.stage)
        try:
            result = validate(args.stage, snap)
            break
        except (AssertionError, KeyError, ValueError, IndexError, OSError) as error:
            changed = snap.changed()
            if args.stage == "preliminary" and changed and attempt < 2:
                continue
            result = dict(status="SNAPSHOT_RACE" if changed else "FAIL", stage=args.stage,
                          error=str(error), changed=changed,
                          generated_utc=datetime.now(timezone.utc).isoformat(),
                          validator_sha256=sha(Path(__file__).read_bytes()))
            break
    OUT.mkdir(exist_ok=True)
    (OUT/"data_validation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "stage", "required_rows", "missing_required_rows", "error") if k in result}))
    if result["status"] not in ("PASS", "PASS_PRELIMINARY"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
