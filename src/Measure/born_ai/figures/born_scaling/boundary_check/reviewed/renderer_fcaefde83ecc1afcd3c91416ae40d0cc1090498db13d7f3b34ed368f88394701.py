"""Presentation-only revision of frozen boundary-check figures.

Reads successful analysis outputs. Does not sample, summarize, or refit data.
Original analysis, figures, and their provenance remain unchanged.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / "analysis/born_scaling/boundary_check/production"
DESTINATION = ROOT / "figures/born_scaling/boundary_check/reviewed"
FROZEN_ANALYZER_SHA256 = "d29682d79e29cacf278ca5b7c93ea758f7a768ff3ad86e35bc2089dcda8e9925"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return path.relative_to(ROOT).as_posix()


def read_csv(path):
    def scalar(value):
        if value == "":
            return None
        if value in ("True", "False"):
            return value == "True"
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value
    with path.open(newline="", encoding="utf-8") as handle:
        return [{key: scalar(value) for key, value in row.items()}
                for row in csv.DictReader(handle)]


def main():
    source = ROOT / "analyze_born_boundary_check.py"
    if digest(source) != FROZEN_ANALYZER_SHA256:
        raise ValueError("Frozen analyzer hash mismatch")
    summary_path = ANALYSIS / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["status"] != "PASS" or summary["stage"] != "production":
        raise ValueError("Requires a successful completed production analysis")
    names = ("means.csv", "fits.json", "local_slopes.csv", "cumulative_diagnostics.csv")
    inputs = {relative(summary_path): digest(summary_path), relative(source): digest(source)}
    for name in names:
        path = ANALYSIS / name
        expected = summary["outputs"][relative(path)]
        if digest(path) != expected:
            raise ValueError(f"Analysis output changed: {path}")
        inputs[relative(path)] = expected
    for name, expected in summary["outputs"].items():
        if name.startswith("figures/born_scaling/boundary_check/"):
            path = ROOT / name
            if digest(path) != expected:
                raise ValueError(f"Original figure changed: {path}")
            inputs[name] = expected

    import analyze_born_boundary_check as frozen
    from matplotlib.figure import Figure

    means = read_csv(ANALYSIS / "means.csv")
    fits = json.loads((ANALYSIS / "fits.json").read_text(encoding="utf-8"))
    slopes = read_csv(ANALYSIS / "local_slopes.csv")
    tables = {"cumulative_diagnostics": read_csv(ANALYSIS / "cumulative_diagnostics.csv")}
    original_savefig = Figure.savefig
    revised = set()

    def reviewed_savefig(figure, path, *args, **kwargs):
        if figure not in revised:
            revised.add(figure)
            axes = figure.axes
            # Move explanatory notes out of the data rectangles.
            for axis in axes:
                for annotation in list(axis.texts):
                    annotation.remove()
            name = Path(path).stem
            if name == "geometry_scaling":
                figure.suptitle(
                    "Common size window: L = 6–36\n"
                    "Solid: log + 1/L; dotted: log. β = 3 fits fail; fitted c is not an identified corner coefficient.",
                    fontsize=11,
                )
                for column, beta in enumerate((0.0, 3.0)):
                    handles, labels = axes[column].get_legend_handles_labels()
                    original_labels = labels.copy()
                    for index, label in enumerate(original_labels):
                        if label.startswith("log +"):
                            # Curve order is p=1/3 then p=1/2; data artists
                            # are ErrorbarContainers and do not start with log.
                            curve_index = sum(text.startswith("log +") for text in original_labels[:index])
                            den = (3, 2)[curve_index]
                            fit = frozen.finite_fit(fits, beta, f"p{den}_group_D", "log_invL")
                            coefficient = fit["coefficient"]["logL"]
                            if beta:
                                labels[index] = (rf"diagnostic $c={coefficient:.3f}$; "
                                                 rf"$\chi^2/\mathrm{{dof}}={fit['reduced_chi_square']:.1f}$")
                            else:
                                labels[index] = rf"log + $1/L$: $c={coefficient:.5f}$"
                    axes[column].legend(handles, labels, fontsize=8.0)
            elif name == "paired_boundary_checks":
                figure.suptitle(
                    "Paired contrasts from the same retained states\n"
                    "Top left: dotted = constant, dashed = constant + 1/L. Both β = 3 fits fail; drift alone is not residual area law.",
                    fontsize=11,
                )
            elif name == "fit_and_sampling_diagnostics":
                figure.suptitle(
                    "Finite-window fit mismatch and sampling checks are separate\n"
                    "Missing fits have fewer than three residual degrees of freedom; no threshold was lowered.",
                    fontsize=11,
                )
                axes[0].set_title(r"$\beta=3$: all shown log / log + 1/L fits fail")
            else:
                raise ValueError(f"Unexpected figure: {name}")
        return original_savefig(figure, path, *args, **kwargs)

    if DESTINATION.exists() and any(DESTINATION.iterdir()):
        raise FileExistsError("Reviewed output directory must be new and empty")
    try:
        Figure.savefig = reviewed_savefig
        outputs = frozen.make_figures(means, fits, slopes, tables, DESTINATION)
    finally:
        Figure.savefig = original_savefig
    renderer_sha = digest(Path(__file__))
    snapshot = DESTINATION / f"renderer_{renderer_sha}.py"
    snapshot.write_bytes(Path(__file__).read_bytes())
    inputs[relative(Path(__file__))] = renderer_sha
    receipt = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "role": "Presentation-only revision using verified saved analysis; no resampling, statistics, or refitting",
        "changes": [
            "Moved notes outside data panels so they do not obscure points",
            "Marked beta=3 fit mismatch directly on the geometry figure",
            "Displayed beta=0 coefficients to five decimals, explicitly restricted to L=6–36",
        ],
        "inputs_sha256": inputs,
        "outputs_sha256": {relative(path): digest(path) for path in outputs + [snapshot]},
        "frozen_analysis_unchanged": digest(summary_path) == inputs[relative(summary_path)],
    }
    receipt_path = DESTINATION / "presentation_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"receipt": relative(receipt_path), "outputs": len(outputs)}, indent=2))


if __name__ == "__main__":
    main()
