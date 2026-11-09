"""Freeze old-data-only beta=3 extrapolations before new production means.

No Monte Carlo and no reads from the new pilot/production directories. Existing
outputs are verified, never overwritten. Predictions are conditional on two
three-parameter models fit to only five old sizes (two residual degrees of
freedom); their uncertainty omits model bias and uncertainty in estimated MCSE.
"""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "ai/analysis_beta_patterns/audit_dependencies"))
import mpmath as mp

MODELS = {"D_log_invL": ("intercept", "logL", "invL"),
          "D_linear_log_diagnostic": ("intercept", "L", "logL")}
OLD_SIZES = [12, 15, 18, 21, 24]
NEW_SIZES = [27, 30, 36, 42]
FIELDS = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    assert not path.exists(), f"Refuse overwrite: {path}"
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def design(sizes, terms):
    return mp.matrix([[{"intercept": mp.mpf(1), "L": mp.mpf(L),
                        "logL": mp.log(L), "invL": mp.mpf(1)/L}[term]
                       for term in terms] for L in sizes])


def main():
    receipt_path = OUT / "summary.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert receipt["status"] == "FROZEN"
        for filename, expected in receipt["files"].items():
            assert sha(ROOT / filename) == expected, filename
        print("PASS: existing immutable holdout artifact verified; nothing rewritten")
        return
    protocol_path = ROOT / "analysis/born_scaling/beta3_large_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert protocol["beta"] == 3. and protocol["mu"] == 1e-8
    assert protocol["primary_large_size_window_L_min"] == 18
    assert protocol["initial_candidate_sizes"] == [27,30,36] and protocol["optional_size"] == 42
    prod_path = ROOT / "analysis/born_scaling/production/summary.json"
    assert sha(prod_path) == "edb5db7350274bd779d6cc30c0396a8fce6ad6077c8a574c0ebd525ee951dea5"
    prod = json.loads(prod_path.read_text(encoding="utf-8"))
    points_path = prod_path.parent / "means.csv"
    assert sha(points_path) == prod["files"][relative(points_path)]
    selected = sorted((r for r in read_csv(points_path) if float(r["beta"]) == 3.
                       and int(r["L"]) in OLD_SIZES), key=lambda r:int(r["L"]))
    assert [int(r["L"]) for r in selected] == OLD_SIZES
    old_fits_path = ROOT / "analysis/born_scaling/linear_crossover/fits.csv"
    old_summary_path = old_fits_path.parent / "summary.json"
    old_summary = json.loads(old_summary_path.read_text(encoding="utf-8"))
    assert sha(old_fits_path) == old_summary["files"][relative(old_fits_path)]
    old_fits = {r["model"]:r for r in read_csv(old_fits_path)
                if float(r["beta"]) == 3. and int(r["L_min"]) == 12 and r["model"] in MODELS}
    predictions, parameter_rows, residual_rows, covariance_rows, influence_rows = [], [], [], [], []
    models = {}
    maximum_old_fit_difference = mp.mpf(0)
    with mp.workdps(80):
        y = mp.matrix([mp.mpf(r["D_mean"]) for r in selected])
        se = [mp.mpf(r["D_MCSE"]) for r in selected]
        V = mp.diag([s*s for s in se])
        W = mp.diag([1/(s*s) for s in se])
        for name, terms in MODELS.items():
            X = design(OLD_SIZES, terms)
            A = mp.matrix([[X[i,j]/se[i] for j in range(len(terms))] for i in range(len(se))])
            B = mp.matrix([y[i]/se[i] for i in range(len(se))])
            coefficient, _ = mp.qr_solve(A, B)
            coefficient_covariance = (A.T*A)**-1
            coefficient_map = coefficient_covariance * X.T * W
            Z = design(NEW_SIZES, terms)
            prediction = Z*coefficient
            influence = Z*coefficient_map
            prediction_covariance = influence*V*influence.T
            direct_covariance = Z*coefficient_covariance*Z.T
            assert max(abs(v) for v in prediction_covariance-direct_covariance) < mp.mpf("1e-65")
            assert max(abs(v) for v in prediction-influence*y) < mp.mpf("1e-65")
            residual = y-X*coefficient
            chi = sum((residual[i]/se[i])**2 for i in range(len(se)))
            for i,term in enumerate(terms):
                difference = abs(coefficient[i]-mp.mpf(old_fits[name][f"coefficient_{term}"]))
                maximum_old_fit_difference = max(maximum_old_fit_difference,difference)
                assert difference < mp.mpf("2e-11")
                for j,term2 in enumerate(terms):
                    parameter_rows.append(dict(model=name,term_i=term,term_j=term2,
                                               covariance=float(coefficient_covariance[i,j]),
                                               covariance_decimal80=mp.nstr(coefficient_covariance[i,j],80)))
            for i,L in enumerate(OLD_SIZES):
                residual_rows.append(dict(model=name,L=L,observed=float(y[i]),MCSE=float(se[i]),
                                          fitted=float(y[i]-residual[i]),residual=float(residual[i]),
                                          standardized_residual=float(residual[i]/se[i])))
            for i,L in enumerate(NEW_SIZES):
                predictions.append(dict(model=name,beta=3.,mu=1e-8,L=L,
                                        optional_size=L==42,D_prediction=float(prediction[i]),
                                        prediction_parameter_MCSE=float(mp.sqrt(prediction_covariance[i,i])),
                                        D_prediction_decimal80=mp.nstr(prediction[i],80),
                                        uncertainty_role="Old-parameter uncertainty only; excludes new MCSE and model bias"))
                for j,old_L in enumerate(OLD_SIZES):
                    influence_rows.append(dict(model=name,prediction_L=L,old_L=old_L,
                                               influence=float(influence[i,j]),
                                               influence_decimal80=mp.nstr(influence[i,j],80)))
            models[name] = dict(terms=list(terms),old_sizes=OLD_SIZES,n_points=5,n_parameters=3,residual_dof=2,
                                coefficient={t:float(coefficient[i]) for i,t in enumerate(terms)},
                                coefficient_decimal80={t:mp.nstr(coefficient[i],80) for i,t in enumerate(terms)},
                                chi_square=float(chi),reduced_chi_square=float(chi/2),
                                covariance=coefficient_covariance,influence=influence)
        for name_a,model_a in models.items():
            for name_b,model_b in models.items():
                cross=model_a["influence"]*V*model_b["influence"].T
                for i,L_i in enumerate(NEW_SIZES):
                    for j,L_j in enumerate(NEW_SIZES):
                        covariance_rows.append(dict(model_i=name_a,L_i=L_i,model_j=name_b,L_j=L_j,
                                                    prediction_covariance=float(cross[i,j]),
                                                    prediction_covariance_decimal80=mp.nstr(cross[i,j],80)))
        for model in models.values():
            del model["covariance"];del model["influence"]
    points=[{key:r[key] for key in ("L","beta","mu","D_mean","D_MCSE")} for r in selected]
    tables={"old_points.csv":points,"old_fit_residuals.csv":residual_rows,
            "predictions.csv":predictions,"parameter_covariance.csv":parameter_rows,
            "prediction_covariance.csv":covariance_rows,"prediction_influence.csv":influence_rows}
    for name,table in tables.items():
        write_csv(OUT/name,table)
    source_paths=[Path(__file__),protocol_path,prod_path,points_path,old_summary_path,old_fits_path,
                  ROOT/"analysis/born_scaling_production_validation.json",
                  ROOT/"analysis/born_scaling/linear_crossover/validation.json"]
    files={relative(p):sha(p) for p in source_paths+[OUT/name for name in tables]}
    receipt=dict(status="FROZEN",created_utc=datetime.now(timezone.utc).isoformat(),decimal_precision=80,
                 scope="Old L12-24 beta3 data only; no new-size pilot or production observation was read",
                 beta=3.,mu=1e-8,old_sizes=OLD_SIZES,prediction_sizes=NEW_SIZES,
                 models=models,maximum_existing_coefficient_difference=float(maximum_old_fit_difference),
                 future_test={
                     "per_model":"For completed preregistered new sizes, residual=y_new-prediction; V_total=diag(new_MCSE^2)+corresponding block of frozen prediction_covariance. Report residual^T V_total^-1 residual without refitting old coefficients.",
                     "degrees_of_freedom":"Number of new size means tested, because coefficients were fixed from independent old data; chi-square is an approximate diagnostic conditional on estimated MCSE and model adequacy.",
                     "missing_optional_size":"L42 is an optional cost-driven size; if not run, take the corresponding covariance submatrix. Do not select test sizes by their observed agreement.",
                     "cross_model_dependence":"Both predictions use the same five old means. Full cross-model covariance is supplied; the two predictive tests also share the new data, so their statistics are not independent evidence.",
                     "sequence":"Score frozen predictions before examining refitted combined-size models; keep both results with distinct interpretations."},
                 limitations=["Each old model had only five points and two residual degrees of freedom; its good fit is not evidence for an asymptotic law.",
                              "Prediction intervals omit model/finite-size bias and uncertainty in the estimated old/new MCSE.",
                              "A failed extrapolation identifies failure of this fixed finite-window continuation, not a proof against every logarithmic or linear asymptotic scenario.",
                              "Four predictions in one model share three fitted coefficients; the parameter-only covariance has rank at most three. New independent MCSE must be added for the future predictive test."],
                 files=files)
    with receipt_path.open("x",encoding="utf-8") as stream:
        json.dump(receipt,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write("\n")
    print(json.dumps(dict(status="FROZEN",models=2,predictions=len(predictions),covariance_entries=len(covariance_rows),
                         summary_sha256=sha(receipt_path)),indent=2))


if __name__=="__main__":
    main()
