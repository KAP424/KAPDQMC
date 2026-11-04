"""Render the Ref calibration from recorded fits; never select or refit data."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "analysis_ref_calibration"
OUT = ROOT / "figures_ref_calibration"
ORDER = ["uniform_A", "uniform_B", "one_B_2", "one_B_3", "two_B_3", "diagonal_B_3",
         "special_matched_a", "special_matched_b", "full_empty_cluster3", "random_half_3",
         "general_matched_a", "general_matched_b"]
BETAS = [0., .5, 1., 2.]
MODEL = "log_plus_inv1"
FOUR = "log_plus_inv1_inv2"
BLUE, ORANGE, BLACK, GREEN = "#1674b8", "#d58422", "#252525", "#398144"


def read(name):
    with (DATA / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def num(r, k, default=math.nan):
    try:
        return float(r[k])
    except (KeyError, TypeError, ValueError):
        return default


def truth(v):
    return str(v).lower() in ("true", "1")


def evaluate(row, x):
    model = row["model"]
    y = num(row, "c") * np.log(x) + num(row, "b")
    if model in (MODEL, FOUR):
        y += num(row, "d", 0.) / x
    if model in ("log_plus_inv2", FOUR):
        y += num(row, "e", 0.) / x**2
    return y


def fitline(ax, row, color, label, style="-", lw=1.6):
    if row is None:
        return
    x = np.geomspace(num(row, "x_min"), num(row, "x_max"), 150)
    ax.plot(np.log(x), evaluate(row, x), style, color=color, lw=lw, label=label)


def scales(ax, xlabel):
    ax.set_xscale("linear")
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$D=(2S_A+S_C)/3-S_B$")
    ax.grid(True, alpha=.2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    points = read("calibration_points.csv")
    fits = read("calibration_fits.csv")
    comparison = read("calibration_comparison.csv")
    complete = [r for r in comparison if math.isfinite(num(r, "c_main108"))]
    if not args.preview:
        assert len(complete) == 48, f"incomplete common scan: {len(complete)}/48"
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white", "figure.facecolor": "white"})
    outputs = []

    def save(fig, stem):
        for suffix in ("png", "pdf"):
            path = OUT / f"{stem}.{suffix}"
            fig.savefig(path, dpi=170, bbox_inches="tight")
            outputs.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        plt.close(fig)

    def getfit(p, b, domain, window, L="", model=MODEL):
        selected = [r for r in fits if r["pattern"] == p and num(r, "beta") == b
                    and r["domain"] == domain and r["window"] == window and r["model"] == model
                    and (domain == "equilibrium" or str(r["L"]) in (str(L), f"{float(L):.1f}"))]
        assert len(selected) <= 1, (p, b, domain, window, L, model)
        return selected[0] if selected else None

    def getcomparison(p, b):
        selected = [r for r in comparison if r["pattern"] == p and num(r, "beta") == b]
        assert len(selected) == 1
        return selected[0]

    def coefficients(ax, p, legend=True):
        rows = [getcomparison(p, b) for b in BETAS]
        ceq = [num(r, "c_eq_old") for r in rows]
        mainc = [num(r, "c_main108") for r in rows]
        fourc = [num(r, "c_four108") for r in rows]
        ax.plot(BETAS, ceq, "o-", color=BLACK, ms=4, lw=1.4, label="Equilibrium, L=18-90")
        ax.plot(BETAS, mainc, "-", color=BLUE, lw=1.2, label=r"Time, L=108, $\theta=2$-6")
        for b, c, r in zip(BETAS, mainc, rows):
            ax.plot(b, c, "s", color=BLUE, ms=6,
                    mfc=BLUE if truth(r.get("stable_diagnostic")) else "white", zorder=5)
        ax.plot(BETAS, fourc, "^--", color=ORANGE, mfc="white", ms=4, lw=1.0,
                label=r"Time + $1/\theta^2$ check")
        ext = num(rows[0], "c_eq_extended")
        if math.isfinite(ext):
            ax.plot(0, ext, "*", ms=10, color=GREEN, mfc="white", mew=1.3,
                    label=r"Equilibrium extended ($\beta=0$)")
        ax.axhline(0, color=".7", lw=.6)
        ax.set_xticks(BETAS)
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel("Finite-window log coefficient")
        ax.grid(True, alpha=.2)
        if legend:
            ax.legend(fontsize=8, loc="best")

    for p in ORDER:
        fig, axes = plt.subplots(4, 2, figsize=(13, 14), constrained_layout=True)
        fig.suptitle(f"{p}: equilibrium / early-time diagnostics\n"
                     "Scale separation not certified; one species; no vertical offsets", fontsize=15)
        for i, beta in enumerate(BETAS):
            pp = [r for r in points if r["pattern"] == p and num(r, "beta") == beta]
            eq = sorted([r for r in pp if math.isinf(num(r, "theta")) and 18 <= num(r, "L") <= 144],
                        key=lambda r:num(r, "L"))
            left, right = axes[i]
            left.plot(np.log([num(r, "L") for r in eq]), [num(r, "D") for r in eq], "o", ms=4,
                      mfc="white", color=BLACK, label="Measured equilibrium D")
            ef = getfit(p, beta, "equilibrium", "18-90")
            fitline(left, ef, BLACK, f"18-90 +1/L: c={num(ef or {}, 'c'):.6g}")
            ext = getfit(p, beta, "equilibrium", "18-144")
            fitline(left, ext, GREEN, f"18-144 +1/L: c={num(ext or {}, 'c'):.6g}", "--")
            left.set_title(rf"$\beta={beta:g}$ : equilibrium")
            scales(left, r"$\ln L$")
            left.legend(fontsize=8, loc="best")
            for L, color, marker in ((72, ".7", "."), (90, "#69aaa6", "o"), (108, BLUE, "s")):
                rr = sorted([r for r in pp if num(r, "L") == L and 2 <= num(r, "theta") <= 8],
                            key=lambda r:num(r, "theta"))
                right.plot(np.log([num(r, "theta") for r in rr]), [num(r, "D") for r in rr], marker,
                           mfc="white", ms=3.8, color=color, label=f"Measured L={L}")
            tf = getfit(p, beta, "time", "2-6", 108)
            ff = getfit(p, beta, "time", "2-6", 108, FOUR)
            fitline(right, tf, BLUE, f"2-6 +1/t: c={num(tf or {}, 'c'):.6g}")
            fitline(right, ff, ORANGE, f"+1/t^2 check: c={num(ff or {}, 'c'):.6g}", "--")
            status = "local checks only" if truth(getcomparison(p, beta).get("stable_diagnostic")) else "estimate NOT stable"
            right.set_title(rf"$\beta={beta:g}$ : time / {status}")
            scales(right, r"$\ln\theta$")
            right.legend(fontsize=8, loc="best")
        save(fig, f"scaling_{p}")
        fig, aa = plt.subplots(1, 2, figsize=(13.5, 5.1), constrained_layout=True)
        coefficients(aa[0], p)
        rr = [getcomparison(p,b) for b in BETAS]
        ref = np.array([num(r,"c_eq_old") for r in rr])
        ax = aa[1]
        ax.axhline(1, color=BLACK, lw=1.2, label="Equilibrium fit = 1")
        for key, color, marker, label in (("c_main108", BLUE, "s", "Time, 2-6 +1/t"),
                                          ("c_four108", ORANGE, "^", "Time +1/t^2")):
            vals = np.array([num(r,key) for r in rr]) / ref
            ax.plot(BETAS, vals, marker=marker, mfc="white", color=color, lw=1.1, label=label)
            if key == "c_main108":
                for b, v, r in zip(BETAS, vals, rr):
                    if truth(r.get("stable_diagnostic")):
                        ax.plot(b, v, marker, color=color)
        ax.set_xticks(BETAS)
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel("Time coefficient / equilibrium coefficient")
        ax.set_title("Relative comparison resolves small coefficients", fontsize=11)
        ax.grid(True, alpha=.2)
        ax.legend(fontsize=8)
        fig.suptitle(p + "\nEarly-window diagnostics: filled = local checks only; scale separation not certified", fontsize=14)
        save(fig, f"coefficients_{p}")

    fig, axes = plt.subplots(3, 4, figsize=(17, 11), constrained_layout=True)
    for ax, p in zip(axes.flat, ORDER):
        coefficients(ax, p, legend=False)
        ax.set_title(p, fontsize=11)
        ax.set_ylabel("Log coefficient", fontsize=9)
        ax.tick_params(labelsize=9)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="outside lower center", fontsize=10)
    fig.suptitle("Ref calibration: equilibrium and time coefficients vs beta\n"
                 "Early-window diagnostics: filled = local checks only, not certified scale separation", fontsize=16)
    save(fig, "coefficient_overview")

    fig, ax = plt.subplots(figsize=(13.5, 6), constrained_layout=True)
    x = np.arange(len(ORDER))
    for shift, key, label, marker, color in ((-.25, "c_eq_old", "Equilibrium L18-90", "o", BLACK),
                                          (-.08, "c_eq_extended", "Equilibrium L18-144", "*", GREEN),
                                          (.09, "c_main108", "Time L108, 2-6 +1/t", "s", BLUE),
                                          (.25, "c_four108", "Time +1/t^2 check", "^", ORANGE)):
        vals = [num(getcomparison(p, 0), key) for p in ORDER]
        ax.plot(x+shift, vals, linestyle="none", marker=marker, color=color, mfc="white", ms=7, label=label)
        if key == "c_main108":
            for xx, val, p in zip(x+shift, vals, ORDER):
                if truth(getcomparison(p, 0).get("stable_diagnostic")):
                    ax.plot(xx, val, marker, color=color, ms=7)
    ax.set_xticks(x, ORDER, rotation=35, ha="right")
    ax.set_ylabel("Finite-window log coefficient")
    ax.set_title(r"$\beta=0$: all 12 configurations, including unresolved estimates")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(axis="y", alpha=.25)
    save(fig, "beta0_all_patterns")

    # Higher-order diagnostics are separate from the fixed-window checks.
    order = read("calibration_order_diagnostics.csv")
    fig, axes = plt.subplots(3, 4, figsize=(17, 11), constrained_layout=True)
    for ax, p in zip(axes.flat, ORDER):
        ax.plot(BETAS, [num(getcomparison(p,b), "c_eq_old") for b in BETAS],
                "o-", color=BLACK, ms=4, label="Equilibrium L18-90")
        for window, model, color, marker, style, label in (
                ("3-8", FOUR, BLUE, "s", "-", "4 parameters, 3-8"),
                ("4-8", FOUR, "#73479b", "D", "--", "4 parameters, 4-8"),
                ("3-8", "log_plus_inv1_inv2_inv3", ORANGE, "^", "-", "5 parameters, 3-8"),
                ("4-8", "log_plus_inv1_inv2_inv3", "#b85656", "v", "--", "5 parameters, 4-8")):
            values = []
            for b in BETAS:
                row = next((r for r in order if r["pattern"] == p and num(r,"beta") == b
                            and r["window"] == window and r["model"] == model), {})
                values.append(num(row, "c_108"))
            ax.plot(BETAS, values, marker=marker, linestyle=style, mfc="white", ms=4,
                    lw=1., color=color, label=label)
        ax.set_title(p, fontsize=11)
        ax.set_xticks(BETAS)
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel("Finite-window log coefficient", fontsize=9)
        ax.axhline(0, color=".8", lw=.6)
        ax.grid(True, alpha=.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="outside lower center", fontsize=10)
    fig.suptitle("Correction-order diagnostics at L=108\n"
                 "Higher order is not automatically more reliable; all values are retained", fontsize=16)
    save(fig, "hierarchy_coefficient_overview")

    late = read("calibration_late_diagnostics.csv")
    late_patterns = [p for p in ORDER if any(r["pattern"] == p for r in late)]
    fig, axes = plt.subplots(3, 3, figsize=(14, 11), constrained_layout=True)
    for ax, p in zip(axes.flat, late_patterns):
        for key, model, color, marker, label in (
                ("c_108", MODEL, ".55", "o", "L108, 3 parameters"),
                ("c_144", MODEL, BLUE, "s", "L144, 3 parameters"),
                ("c_144", FOUR, ORANGE, "^", "L144, 4 parameters")):
            values = []
            for window in ("4-10", "6-12"):
                row = next((r for r in late if r["pattern"] == p and r["window"] == window
                            and r["model"] == model), {})
                values.append(num(row, key))
            ax.plot([0, 1], values, marker=marker, mfc="white", color=color, label=label)
        ax.axhline(num(getcomparison(p,0), "c_eq_extended"), color=BLACK, ls="--", lw=1.,
                   label="Equilibrium L18-144")
        ax.set_xticks([0,1], ["4-10", "6-12"])
        ax.set_xlabel(r"$\theta$ window (7 measured integer points)", fontsize=9)
        ax.set_ylabel("Finite-window log coefficient", fontsize=9)
        ax.set_title(p, fontsize=11)
        ax.grid(True, alpha=.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="outside lower center", fontsize=10)
    fig.suptitle(r"$\beta=0$: delayed-time diagnostics at L108 and L144" + "\n"
                 "Extending time also increases the finite-size requirement", fontsize=16)
    save(fig, "late_beta0_diagnostics")
    sources = {n: hashlib.sha256((DATA/n).read_bytes()).hexdigest() for n in
               ("calibration_points.csv", "calibration_fits.csv", "calibration_comparison.csv",
                "calibration_order_diagnostics.csv", "calibration_late_diagnostics.csv")}
    manifest = dict(partial_render=args.preview, input_sha256=sources, files=outputs,
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    figures=len(outputs)//2, scaling_coordinates="x=ln(L) or ln(theta), linear axes; y=D",
                    note="Plots evaluate recorded fit coefficients at physical L/theta; no refitting or target selection")
    (OUT / "plot_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Rendered {len(outputs)//2} PNG/PDF pairs")


if __name__ == "__main__":
    main()
