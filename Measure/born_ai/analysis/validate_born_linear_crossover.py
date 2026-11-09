"""Independent 80-digit QR audit of the post-hoc linear/log comparison; no MC."""
from pathlib import Path
import csv
import hashlib
import json
import math
import os
import sys
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "ai/analysis_beta_patterns/audit_dependencies"))
import mpmath as mp

DIR = ROOT / "analysis/born_scaling/linear_crossover"
MODELS = {"D_linear": ("intercept","L"), "D_log": ("intercept","logL"),
          "D_log_invL": ("intercept","logL","invL"),
          "D_linear_log_diagnostic": ("intercept","L","logL")}
errors = {}
checks = 0


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def check(name, actual, expected):
    global checks
    error = abs(float(actual)-float(expected))
    errors[name] = max(errors.get(name,0.),error)
    checks += 1
    assert error < 3e-10 + 3e-11*abs(float(expected)), (name,actual,expected,error)


def main():
    summary_path = DIR / "summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "PASS" and summary["fits"] == 48 and summary["local_slopes"] == 24
    assert sha(summary_path) == "fc74de29d66e78415705e03178ba5ef95ccdfd903a5f142979b155508228cae3"
    files = dict(summary["files"])
    for name, expected in files.items():
        assert sha(ROOT/name) == expected, name
    production_path = ROOT / "analysis/born_scaling/production/summary.json"
    assert sha(production_path) == "edb5db7350274bd779d6cc30c0396a8fce6ad6077c8a574c0ebd525ee951dea5"
    production = json.loads(production_path.read_text())
    for name,expected in production["files"].items():
        assert sha(ROOT/name) == expected, name
    points = read(DIR/"points.csv")
    prod_points = {(int(r["L"]),float(r["beta"])):r for r in read(production_path.parent/"means.csv")}
    assert len(points) == 28
    point_map = {}
    for r in points:
        key = (int(r["L"]),float(r["beta"]))
        assert key not in point_map
        point_map[key] = r
        for field in ("D_mean","D_MCSE","mu"):
            assert float(r[field]) == float(prod_points[key][field])
    fits = read(DIR/"fits.csv")
    residuals = {(float(r["beta"]),int(r["L_min"]),r["model"],int(r["L"])):r for r in read(DIR/"residuals.csv")}
    assert len(fits) == 48 and len(residuals) == 288
    assert {(float(r["beta"]),int(r["L_min"]),r["model"]) for r in fits} == {(b,lo,m) for b in (1.5,2.,2.5,3.) for lo in (6,9,12) for m in MODELS}
    with mp.workdps(80):
        for row in fits:
            beta,lower,model = float(row["beta"]),int(row["L_min"]),row["model"]
            sizes = list(range(lower,25,3)); terms = MODELS[model]
            y = mp.matrix([mp.mpf(point_map[(L,beta)]["D_mean"]) for L in sizes])
            se = [mp.mpf(point_map[(L,beta)]["D_MCSE"]) for L in sizes]
            columns = [{"intercept":mp.mpf(1),"L":mp.mpf(L),"logL":mp.log(L),"invL":mp.mpf(1)/L} for L in sizes]
            X = mp.matrix([[col[t] for t in terms] for col in columns])
            A = mp.matrix([[X[i,j]/se[i] for j in range(len(terms))] for i in range(len(sizes))])
            B = mp.matrix([y[i]/se[i] for i in range(len(sizes))])
            coef,_ = mp.qr_solve(A,B)
            covariance = (A.T*A)**-1
            residual = y-X*coef
            chi = sum((residual[i]/se[i])**2 for i in range(len(sizes)))
            dof = len(sizes)-len(terms)
            assert int(row["n_sizes"]) == len(sizes) and int(row["n_parameters"]) == len(terms) and int(row["residual_dof"]) == dof
            assert (row["small_window_caution"]=="True") == (dof<3)
            assert row["comparison_role"] == "posthoc_current_window_diagnostic_only"
            check("chi_square",row["chi_square"],chi)
            check("reduced_chi_square",row["reduced_chi_square"],chi/dof)
            check("rmse",row["residual_rmse"],mp.sqrt(sum(v*v for v in residual)/len(sizes)))
            assert (row["large_fit_residual"]=="True") == (chi/dof>3)
            for j,term in enumerate(terms):
                check("coefficient",row[f"coefficient_{term}"],coef[j])
                check("parameter_MCSE",row[f"MCSE_{term}"],mp.sqrt(covariance[j,j]))
            if "L" in terms and "logL" in terms:
                i,j=terms.index("L"),terms.index("logL")
                check("L_log_correlation",row["correlation_L_logL"],covariance[i,j]/mp.sqrt(covariance[i,i]*covariance[j,j]))
            for i,L in enumerate(sizes):
                r = residuals[(beta,lower,model,L)]
                check("residual",r["residual"],residual[i])
                check("standardized_residual",r["standardized_residual"],residual[i]/se[i])
            matrix=np.array(A.tolist(),dtype=float)
            check("condition",row["weighted_design_condition"],np.linalg.cond(matrix/np.linalg.norm(matrix,axis=0)))
    slopes=read(DIR/"local_slopes.csv")
    assert len(slopes)==24
    for row in slopes:
        beta,left,right=float(row["beta"]),int(row["L_left"]),int(row["L_right"])
        a,b=point_map[(left,beta)],point_map[(right,beta)]
        delta=float(b["D_mean"])-float(a["D_mean"])
        se=math.hypot(float(a["D_MCSE"]),float(b["D_MCSE"]))
        check("linear_slope",row["linear_slope"],delta/(right-left))
        check("linear_slope_MCSE",row["linear_slope_MCSE"],se/(right-left))
        check("log_slope",row["logarithmic_slope"],delta/math.log(right/left))
        check("log_slope_MCSE",row["logarithmic_slope_MCSE"],se/math.log(right/left))
        check("midpoint",row["L_midpoint"],(right+left)/2)
    files[str(summary_path.relative_to(ROOT)).replace('\\','/')]=sha(summary_path)
    files[str(Path(__file__).relative_to(ROOT)).replace('\\','/')]=sha(Path(__file__))
    result=dict(status="PASS",fit_count=48,slopes=24,decimal_precision=80,numeric_comparisons=checks,
                maximum_absolute_differences=errors,production_unchanged=True,
                production_summary_sha256=sha(production_path),supplement_summary_sha256=sha(summary_path),
                scope="Independent recomputation of post-hoc current-window WLS, parameter errors, residuals, and local slopes; no MC or asymptotic inference",
                caveats=["Lmin12 is post hoc; five-size three-parameter fits have only two residual degrees of freedom",
                         "A linear fit over these sizes does not establish a thermodynamic linear law",
                         "Nearby windows/slopes share data; low chi-square is not an independent replication"], files=files)
    (DIR/"validation.json").write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({k:result[k] for k in ("status","fit_count","slopes","numeric_comparisons","maximum_absolute_differences")},indent=2))


if __name__=="__main__":
    main()
