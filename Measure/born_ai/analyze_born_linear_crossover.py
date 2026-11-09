"""Post hoc current-size-window diagnostics; no new MC or production rewrites.

The linear model is compared with the already reported logarithmic models.
Known Monte Carlo errors set WLS weights and parameter errors; residuals never
rescale those errors. A successful small-window fit is not an asymptotic claim.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
from datetime import datetime, timezone

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis/born_scaling/linear_crossover"
FIG = ROOT / "figures/born_scaling/linear_crossover"
BETAS = (1.5, 2.0, 2.5, 3.0)
CUTS = (6, 9, 12)
MODELS = {
    "D_linear": ("intercept", "L"),
    "D_log": ("intercept", "logL"),
    "D_log_invL": ("intercept", "logL", "invL"),
    "D_linear_log_diagnostic": ("intercept", "L", "logL"),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path):
    return path.relative_to(ROOT).as_posix()


def read(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def design(L, model):
    cols = dict(intercept=np.ones_like(L), L=L, logL=np.log(L), invL=1/L)
    return np.column_stack([cols[term] for term in MODELS[model]])


def fit(L, y, se, model):
    """Independent column-scaled weighted QR, with unscaled parameter covariance."""
    X = design(L, model)
    A = X / se[:, None]
    b = y / se
    scale = np.linalg.norm(A, axis=0)
    Q, R = np.linalg.qr(A / scale, mode="reduced")
    if np.linalg.matrix_rank(R) != X.shape[1]:
        raise ValueError("Rank-deficient diagnostic fit")
    coef = np.linalg.solve(R, Q.T @ b) / scale
    transform = np.linalg.solve(R, np.eye(R.shape[0])) / scale[:, None]
    cov = transform @ transform.T
    residual = y - X @ coef
    chisq = float(np.sum((residual / se)**2))
    dof = len(L) - len(coef)
    if dof <= 0:
        raise ValueError("No residual degrees of freedom")
    errors = np.sqrt(np.diag(cov))
    result = dict(n_sizes=len(L), n_parameters=len(coef), residual_dof=dof,
                  chi_square=chisq, reduced_chi_square=chisq/dof,
                  weighted_design_condition=float(np.linalg.cond(A/scale)),
                  residual_rmse=float(np.sqrt(np.mean(residual**2))),
                  large_fit_residual=bool(chisq/dof > 3),
                  small_window_caution=bool(len(L) < len(coef)+3))
    for i, term in enumerate(MODELS[model]):
        result[f"coefficient_{term}"] = float(coef[i])
        result[f"MCSE_{term}"] = float(errors[i])
    if "L" in MODELS[model] and "logL" in MODELS[model]:
        i, j = MODELS[model].index("L"), MODELS[model].index("logL")
        result["correlation_L_logL"] = float(cov[i, j] / errors[i] / errors[j])
    return result, residual


def plots(points, fits, slopes):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(3, 3, figsize=(15.6, 10.4), constrained_layout=True)
    for row, beta in enumerate((2., 2.5, 3.)):
        selected = [r for r in points if r["beta"] == beta]
        L = np.array([r["L"] for r in selected], dtype=float)
        y = np.array([r["D_mean"] for r in selected])
        se = np.array([r["D_MCSE"] for r in selected])
        curve = np.linspace(9, 24, 250)
        for column in (0, 1):
            ax = axes[row, column]
            ax.errorbar(L if column == 0 else np.log(L), y, yerr=se,
                        fmt="o", color="#286B9C", mfc="white", ms=7,
                        capsize=2.5, label="Production means / MCSE", zorder=4)
            for model, color, style, label in (
                ("D_linear", "#C47528", "--", "aL+b"),
                ("D_log", "#29847E", ":", "c ln L+b")):
                f = next(r for r in fits if r["beta"] == beta and
                         r["model"] == model and r["L_min"] == 9)
                coefs = np.array([f[f"coefficient_{t}"] for t in MODELS[model]])
                ax.plot(curve if column == 0 else np.log(curve), design(curve, model) @ coefs,
                        color=color, ls=style, lw=1.8,
                        label=f"{label}, chi2/dof={f['reduced_chi_square']:.2f}")
            ax.set(xlabel="L" if column == 0 else r"$\ln L$", ylabel="Born mean D",
                   title=f"beta={beta:g} | {'linear' if column == 0 else 'logarithmic'} horizontal axis")
            ax.grid(alpha=.17)
            ax.legend(fontsize=8, loc="upper left")
        ss = [r for r in slopes if r["beta"] == beta]
        ax = axes[row, 2]
        ax.errorbar([r["L_midpoint"] for r in ss], [r["linear_slope"] for r in ss],
                    yerr=[r["linear_slope_MCSE"] for r in ss], color="#286B9C",
                    marker="o", mfc="white", ms=7, capsize=2.5, label="Adjacent-size differences")
        f = next(r for r in fits if r["beta"] == beta and r["model"] == "D_linear" and r["L_min"] == 9)
        ax.axhline(f["coefficient_L"], color="#C47528", ls="--", label="Linear fit slope, L=9–24")
        ax.set(xlabel=r"Interval midpoint $(L_1+L_2)/2$",
               ylabel=r"$(D_2-D_1)/(L_2-L_1)$", title=f"beta={beta:g} | raw linear slopes")
        ax.grid(alpha=.17)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("Post hoc comparison in the measured size window | no asymptotic linear-scaling claim", fontsize=14, y=.995)
    fig.get_layout_engine().set(rect=(0, .085, 1, .84))
    fig.text(.5, .038, "Curves: two-parameter WLS fits to L=9–24 only; all raw L=6–24 points retained. Three-parameter and Lmin=6/12 checks are in the tables.",
             ha="center", fontsize=10)
    fig.text(.5, .015, "Errors: estimated production MCSE (mu=1e-8). Adjacent slopes share endpoints and are correlated; lines do not establish a limiting slope.",
             ha="center", fontsize=10)
    paths = []
    for suffix in ("png", "pdf"):
        path = FIG / f"linear_crossover.{suffix}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def main():
    production_path = ROOT / "analysis/born_scaling/production/summary.json"
    validation_path = ROOT / "analysis/born_scaling_production_validation.json"
    production = json.loads(production_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if production["status"] != "PASS" or validation["status"] != "PASS":
        raise ValueError("Complete, independently audited production is required")
    dependencies = dict(production["files"])
    dependencies[rel(production_path)] = sha(production_path)
    dependencies[rel(validation_path)] = sha(validation_path)
    for path, digest in dependencies.items():
        if sha(ROOT/path) != digest:
            raise ValueError(f"Changed production dependency: {path}")
    points = []
    for r in read(ROOT / "analysis/born_scaling/production/means.csv"):
        if r["grid"] == "main" and float(r["beta"]) in BETAS:
            if r["sampling_screen_pass"] != "True" or r["deterministic"] != "False":
                raise ValueError("Unexpected production sampling status")
            points.append(dict(L=int(r["L"]), beta=float(r["beta"]), mu=float(r["mu"]),
                               D_mean=float(r["D_mean"]), D_MCSE=float(r["D_MCSE"])))
    points.sort(key=lambda r:(r["beta"],r["L"]))
    fits, residuals, slopes = [], [], []
    for beta in BETAS:
        selected = [r for r in points if r["beta"] == beta]
        if [r["L"] for r in selected] != [6, 9, 12, 15, 18, 21, 24]:
            raise ValueError("Wrong measured size grid")
        for lower in CUTS:
            chosen = [r for r in selected if r["L"] >= lower]
            L = np.array([r["L"] for r in chosen], dtype=float)
            y = np.array([r["D_mean"] for r in chosen])
            se = np.array([r["D_MCSE"] for r in chosen])
            for model in MODELS:
                base = dict(beta=beta, mu=1e-8, model=model, L_min=lower, L_max=24,
                            window_role="new_posthoc_window" if lower == 12 else "previously_reported_window",
                            comparison_role="posthoc_current_window_diagnostic_only")
                result, delta = fit(L, y, se, model)
                fits.append(dict(**base, **result))
                residuals.extend(dict(**base, L=int(l), observed=float(v), MCSE=float(s),
                                      residual=float(d), standardized_residual=float(d/s))
                                 for l, v, s, d in zip(L, y, se, delta))
        for left, right in zip(selected[:-1], selected[1:]):
            dl = right["L"]-left["L"]
            dlog = math.log(right["L"]/left["L"])
            dy = right["D_mean"]-left["D_mean"]
            error = math.hypot(left["D_MCSE"], right["D_MCSE"])
            slopes.append(dict(beta=beta, L_left=left["L"], L_right=right["L"],
                               L_midpoint=(left["L"]+right["L"])/2,
                               linear_slope=dy/dl, linear_slope_MCSE=error/dl,
                               logarithmic_slope=dy/dlog, logarithmic_slope_MCSE=error/dlog,
                               dependence="Independent size runs; adjacent intervals share endpoints"))
    # Independently reproduce old-window fits before introducing the new linear model.
    old = read(ROOT / "analysis/born_scaling/production/fits.csv")
    checks, maximum = 0, 0.
    for r in fits:
        if r["L_min"] == 12 or r["model"] == "D_linear":
            continue
        reference = next(x for x in old if float(x["beta"]) == r["beta"] and
                         int(x["L_min"]) == r["L_min"] and x["model"] == r["model"])
        for key in ["reduced_chi_square"]+[f"{prefix}_{term}" for prefix in ("coefficient", "MCSE") for term in MODELS[r["model"]]]:
            a, b = r[key], float(reference[key])
            if not np.isclose(a, b, atol=2e-10, rtol=2e-10):
                raise ValueError(f"Independent WLS disagrees: {r['beta'],r['model'],r['L_min'],key}")
            maximum = max(maximum, abs(a-b)); checks += 1
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, rows in (("points.csv", points), ("fits.csv", fits), ("residuals.csv", residuals), ("local_slopes.csv", slopes)):
        path = OUT/name
        write(path, rows); paths.append(path)
    paths += plots(points, fits, slopes)
    dependencies.update({rel(p):sha(p) for p in paths})
    dependencies[rel(Path(__file__).resolve())] = sha(Path(__file__).resolve())
    summary = dict(status="PASS", created_utc=datetime.now(timezone.utc).isoformat(),
                   role="Post hoc supplementary current-size-window analysis; production data/results unchanged",
                   betas=BETAS, L_min_values=CUTS, L_max=24, fits=len(fits), local_slopes=len(slopes),
                   independent_existing_fit_checks=checks, maximum_existing_fit_difference=maximum,
                   conventions=dict(weights="1 / production MCSE squared; no residual variance rescaling",
                                    parameters_and_dof="Lmin6/9/12 has 7/6/5 sizes; two-parameter models df5/4/3; three-parameter models df4/3/2",
                                    new_window="Lmin12 is explicitly post hoc; 5-point three-parameter fits have only 2 residual degrees of freedom",
                                    comparison="Linear and log-only have equal parameter counts. Three-parameter models are more flexible; lower residuals alone do not establish the asymptotic form.",
                                    uncertainty="Parameter MCSE excludes model/window uncertainty; overlapping windows and adjacent slopes are correlated",
                                    interpretation="This analysis cannot establish an L to infinity linear law at finite mu/beta or reject a later crossover."),
                   files=dependencies)
    (OUT/"summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
