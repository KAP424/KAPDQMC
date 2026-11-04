"""Finalize the combined data audit without changing the frozen pilot auditor.

The pilot receipt already binds validate_beta0_study_data.py. Keep that file
unchanged and add a direct byte-for-byte prefix check here, so an append cannot
silently replace or reorder any previously certified observation or ledger row.
"""
import hashlib
import json
import pathlib
import sys

import validate_beta0_study_data as audit


def main():
    root = pathlib.Path(__file__).resolve().parent
    directory = root / "analysis_beta0_study"
    prior = json.loads((directory / "pilot_data_validation.json").read_text(encoding="utf-8"))
    assert prior["status"] == "PASS" and prior["stage"] == "pilot"
    prefix_proofs = {}
    for name, relative_snapshot in prior["snapshots"].items():
        frozen = (root / relative_snapshot).read_bytes()
        live = (root / "data" / name).read_bytes()
        assert live.startswith(frozen), f"certified pilot prefix changed: {name}"
        prefix_proofs[name] = {"frozen_bytes": len(frozen),
                              "frozen_sha256": hashlib.sha256(frozen).hexdigest()}
    sys.argv = [str(root / "validate_beta0_study_data.py"), "--stage", "final"]
    audit.main()
    path = directory / "final_data_validation.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    result["pilot_prefix_proofs"] = prefix_proofs
    result["files"][pathlib.Path(__file__).name] = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print("PASS: certified pilot observation and ledger bytes are unchanged prefixes")


if __name__ == "__main__":
    main()
