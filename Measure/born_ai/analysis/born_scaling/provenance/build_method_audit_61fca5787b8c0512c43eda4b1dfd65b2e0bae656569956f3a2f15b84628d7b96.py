"""Index the independent state/method/final-production checks after completion.

Requires the independent final audit. The pre-production receipt is preserved
as an immutable snapshot before replacing its obsolete pending statements.
"""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import tomllib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/born_scaling_method_audit.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    import os
    return Path(os.path.relpath(path, ROOT)).as_posix()


def main():
    files = {}
    receipts = {}
    final_audit_path = ROOT / "analysis/born_scaling_production_validation.json"
    if not final_audit_path.exists():
        raise RuntimeError("Run the independent final production audit first; old review is not overwritten")
    final_audit = json.loads(final_audit_path.read_text(encoding="utf-8"))
    assert final_audit["status"] == "PASS" and final_audit["completed_cases"] == 60
    receipt_paths = (
        ROOT / "analysis/born_scaling_state_audit.toml",
        ROOT / "data/born_scaling/exact_L3_mu1e-8.toml",
        ROOT / "analysis/born_scaling_mass_filter_sensitivity.toml",
        ROOT / "data/born_scaling/beta0_mass_large_sizes.toml",
    )
    checked_hashes = 0
    for path in receipt_paths:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        assert data["status"] == "PASS", path
        for source, expected in data["source_hashes"].items():
            source_path = Path(source)
            assert sha(source_path) == expected, source
            files[relative(source_path)] = expected
            checked_hashes += 1
        for filename, expected in data["files"].items():
            source_path = ROOT / filename if "/" in filename else path.parent / filename
            assert sha(source_path) == expected, source_path
            files[relative(source_path)] = expected
            checked_hashes += 1
        files[relative(path)] = sha(path)
        receipts[relative(path)] = {k: data[k] for k in ("status", "scope") if k in data}

    # Reuse the exact analyzer that generated the validated final outputs,
    # rather than depending on a mutable latest entrypoint.
    analysis_path = ROOT / final_audit["analysis_source"]["source_snapshot"]
    assert sha(analysis_path) == final_audit["analysis_source"]["executed_source_sha256"]
    runner_path = ROOT / "run_born_scaling.jl"
    spec = importlib.util.spec_from_file_location("born_analysis_method_check", analysis_path)
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    baseline = list(csv.DictReader((ROOT / "analysis/born_scaling_state_audit.csv").open()))
    baseline = [r for r in baseline if r["state"] == "staggered_mu1e-8"]
    baseline += list(csv.DictReader((ROOT / "data/born_scaling/beta0_mass_large_sizes.csv").open()))
    sizes = np.array([float(r["L"]) for r in baseline])
    values = np.array([float(r["D"]) for r in baseline])
    reference = list(csv.DictReader((ROOT / "data/born_scaling/beta0_mass_combined_window_fits.csv").open()))
    max_coefficient_error = 0.
    fit_checks = 0
    for row in reference:
        if row["Lmax"] != "36":
            continue
        model = "D_log_invL" if row["inverse_order"] == "1" else "D_log"
        actual = analysis.finite_size_fit(sizes, values, np.zeros(len(sizes)), model, L_min=float(row["Lmin"]))
        assert actual["status"] == "finite_window_diagnostic_only"
        error = abs(actual["coefficient"]["logL"] - float(row["c_window"]))
        assert error < 1e-10
        max_coefficient_error = max(max_coefficient_error, error)
        fit_checks += 1
    for path in (analysis_path, runner_path):
        files[relative(path)] = sha(path)

    for path in (final_audit_path, ROOT / "analysis/born_scaling_auxiliary_j.json",
                 ROOT / "analysis/born_scaling_strong_measurement.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["status"].startswith("PASS"), path
        for filename, expected in data["files"].items():
            assert sha(ROOT / filename) == expected, filename
            files[filename] = expected
            checked_hashes += 1
        files[relative(path)] = sha(path)
        receipts[relative(path)] = {k: data[k] for k in ("status", "scope") if k in data}
    for stage in ("pilot", "smallcheck"):
        path = ROOT / f"analysis/born_scaling/{stage}/summary.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["status"] == "PASS"
        for filename, expected in data["files"].items():
            assert sha(ROOT / filename) == expected, filename
            files[filename] = expected
            checked_hashes += 1
        files[relative(path)] = sha(path)
    provenance = ROOT / "analysis/born_scaling/provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        old_digest = sha(OUT)
        old_path = provenance / f"method_audit_previous_{old_digest}.json"
        if old_path.exists():
            assert sha(old_path) == old_digest
        else:
            old_path.write_bytes(OUT.read_bytes())
    code_sha = sha(Path(__file__))
    code_snapshot = provenance / f"build_method_audit_{code_sha}.py"
    if code_snapshot.exists():
        assert sha(code_snapshot) == code_sha
    else:
        code_snapshot.write_bytes(Path(__file__).read_bytes())
    files[relative(code_snapshot)] = code_sha

    result = {
        "status": "PASS",
        "scope": "Independent deterministic state, finite-filter, exact-enumeration and beta0 size checks; completed raw-production/statistics/80-digit-fit audit. This is NOT a proof of MC tail coverage or asymptotic scaling.",
        "stage": "final_production_review",
        "index_builder_source_snapshot": relative(code_snapshot),
        "analysis_source": final_audit["analysis_source"],
        "checked_receipt_hash_entries": checked_hashes,
        "receipts": receipts,
        "state_audit": {"sizes": list(range(3, 25, 3)), "prescriptions_per_size": 4,
            "main_mu": 1e-8, "sensitivity_mu": 1e-9,
            "mu_sensitivity_max_D": 9.838702297315649e-9,
            "mu_sensitivity_max_S_A": 3.396195147331582e-8},
        "finite_filter_checks": {"sizes": [6, 12, 18], "betas": [.5, 1.5, 3.],
            "fixed_configurations_per_L": 11, "paired_comparisons": 99,
            "not_a_Born_average": True},
        "exact_reference": {"L": 3, "mu": 1e-8, "betas": [.5, 1.5, 3.],
            "configurations_per_beta": 48620, "observables": ["S_A", "S_B", "S_C", "D"],
            "includes": ["means", "full covariance", "per-outcome probabilities and entropy", "weighted-probability tails"]},
        "beta0_extension": {"new_sizes": [27, 30, 36], "positive_beta_data_created": False},
        "independent_fit_crosscheck": {"checks": fit_checks, "reference": "Separate Julia QR least-squares on deterministic D",
            "maximum_log_coefficient_difference": max_coefficient_error},
        "static_runner_review": {
            "same_measurement_beta_as_target": True,
            "fixed_half_cardinality_preserved": True,
            "rejected_states_retained_in_regular_trace": True,
            "geometry": "A=(L/3)xL, B=(L/3)x(2L/3), C=Lx(L/3), both sites per cell",
            "D_formed_per_trajectory": True,
            "records_include_burnin": True,
            "start_labels": "New runner uniform_A means E=all A (suppressed A), unlike older occupied-target naming. This is a labeling issue, not a transition-kernel error.",
        },
        "completed_production_analysis_review": {
            "receipt": relative(final_audit_path),
            "completed_cases": final_audit["completed_cases"],
            "completed_chains": final_audit["completed_chains"],
            "retained_observations": final_audit["retained_observations"],
            "sweeps_retained": "Exactly sweeps 257 through each frozen job end; no initial/pilot records",
            "statistics": "Independently recomputed raw means, rank/folded Rhat, bulk/tail/raw ESS, MCSE, blocks, region covariance, prefixes and tail counts",
            "fits": "Independent 80-decimal QR coefficients and non-residual-rescaled parameter MCSE; all prescribed models/cutoffs retained",
            "sampling_selection": "No expected equilibrium coefficient or desired fit quality enters sample allocation or point selection",
        },
        "limitations_after_final_review": [
            "ESS/Rhat screens do not certify rare-tail coverage. Absolute MCSE targets do not establish relative precision for small means.",
            "Reported parameter MCSE is conditional on estimated point errors and excludes finite-size/model uncertainty; no universal c is certified.",
            "The fixed-mu finite-size window cannot be extrapolated to a massless thermodynamic limit at constant nonzero mu.",
            "Auxiliary-j normalization and fixed-L strong-filter expansions do not determine the large-L corner coefficient.",
        ],
        "files": files,
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"PASS: {checked_hashes} receipt hash entries, {fit_checks} independent fit comparisons; {OUT}")


if __name__ == "__main__":
    main()
