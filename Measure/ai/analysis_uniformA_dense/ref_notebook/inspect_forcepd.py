"""Read saved ForcePD notebook/data and export its images, without executing cells.

Only files in this script's directory are written. Independent NumPy fitting
checks the stored CSV; neither the notebook nor the Julia generator is run.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
AI = HERE.parents[1]
BASE = AI.parent / "Force/victory/A_B"
NOTEBOOK = BASE / "ForcePD.ipynb"
SOURCES = [NOTEBOOK, BASE / "test.jl", BASE / "EA.csv", BASE / "EB.csv",
           BASE / "\u0394E.csv", BASE / "figures/log_scaling_fit.csv",
           AI.parents[1] / "public/GF.jl", AI / "data/ground_observations_baseline.csv",
           AI / "fermion_experiment.jl", AI / "real_entropy_extension.jl",
           AI / "strong_measurement_controls.jl"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(name, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (HERE / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_grid(path):
    raw = np.loadtxt(path, delimiter=",", ndmin=2)
    rows = {int(row[0]): row[1:] for row in raw[1:]}
    sizes = sorted(rows)
    return raw[0, 1:], np.array(sizes), np.array([rows[size] for size in sizes]), len(raw)-1-len(rows)


def main():
    before = {str(path): sha(path) for path in SOURCES}
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    image_names = {3: "saved_entropy_overview", 5: "saved_log_scaling_examples", 7: "saved_alpha_vs_beta"}
    cells, images = [], []
    for i, cell in enumerate(notebook["cells"]):
        source = "".join(cell.get("source", []))
        texts = []
        for j, output in enumerate(cell.get("outputs", [])):
            data = output.get("data", {})
            if "image/png" in data:
                encoded = data["image/png"]
                if isinstance(encoded, list):
                    encoded = "".join(encoded)
                filename = image_names.get(i, f"saved_cell_{i}_output_{j}") + ".png"
                image_path = HERE / filename
                image_path.write_bytes(base64.b64decode(encoded))
                images.append(dict(cell=i, output=j, path=filename, sha256=sha(image_path),
                                   source="Exact saved notebook image/png output; no rerendering."))
            if "text/plain" in data:
                texts.append("".join(data["text/plain"]))
            if output.get("output_type") == "stream":
                texts.append("".join(output.get("text", [])))
        cells.append(dict(index=i, cell_type=cell["cell_type"], execution_count=cell.get("execution_count"),
                          source=source, saved_text_output=texts))
    (HERE / "notebook_cells.json").write_text(json.dumps(cells, indent=2, ensure_ascii=False), encoding="utf-8")
    text = "\n\n".join(f"CELL {row['index']} ({row['cell_type']}), saved execution={row['execution_count']}\n"
                         + row["source"] + "\nSAVED TEXT OUTPUT:\n" + "\n".join(row["saved_text_output"]) for row in cells)
    (HERE / "notebook_source_and_saved_text.txt").write_text(text, encoding="utf-8")

    arrays = [read_grid(BASE / name) for name in ("EA.csv", "EB.csv", "\u0394E.csv")]
    beta, sizes, sa, _ = arrays[0]
    assert all(np.array_equal(beta, b) and np.array_equal(sizes, ls) for b, ls, _, _ in arrays)
    sb, delta = arrays[1][2], arrays[2][2]
    identity_error = float(np.max(abs(delta-sa+sb)))
    assert identity_error < 1e-10
    x = np.log(sizes)
    alpha, intercept = np.polyfit(x, delta, 1)
    rmse = np.sqrt(np.mean((delta - (x[:, None]*alpha+intercept))**2, axis=0))
    saved = np.loadtxt(BASE / "figures/log_scaling_fit.csv", delimiter=",", skiprows=1)
    reproduced = np.column_stack((beta, alpha, intercept, rmse))
    fit_difference = float(np.max(abs(reproduced-saved)))
    assert fit_difference < 1e-12
    write_csv("recomputed_saved_fit.csv", [dict(beta=float(b), alpha=float(a), intercept=float(c),
        rmse=float(r), L_min=int(sizes.min()), L_max=int(sizes.max()), n_sizes=len(sizes),
        model="alpha*ln(L)+intercept", normalization="single_species_natural_log_Renyi2")
        for b, a, c, r in zip(beta, alpha, intercept, rmse)])

    comparisons=[]
    with (AI / "data/ground_observations_baseline.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["pattern"] != "uniform_A" or not math.isinf(float(row["theta"])):
                continue
            size, strength = int(row["L"]), float(row["beta"])
            if size not in sizes:
                continue
            match = np.flatnonzero(np.isclose(beta, strength, atol=1e-12, rtol=0))
            if len(match) != 1:
                continue
            i, j = int(np.flatnonzero(sizes==size)[0]), int(match[0])
            current_sa, current_sc, current_sb, current_d = [float(row[k]) for k in ("S_A", "S_C", "S_B", "D")]
            comparisons.append(dict(L=size, beta=strength, saved_S_A=float(sa[i,j]), saved_S_B=float(sb[i,j]),
                saved_delta=float(delta[i,j]), current_S_A=current_sa, current_S_C=current_sc,
                current_S_B=current_sb, current_D=current_d,
                saved_over_current_D=float(delta[i,j])/current_d,
                saved_minus_current_D=float(delta[i,j])-current_d,
                saved_minus_twice_current_D=float(delta[i,j])-2*current_d,
                current_strip_orientation_difference=current_sa-current_sc,
                saved_minus_current_S_A=float(sa[i,j])-current_sa,
                source_row=row["source_row"], source_file=row["source_file"]))
    write_csv("normalization_comparison.csv", comparisons)
    assert comparisons
    ratios=[r["saved_over_current_D"] for r in comparisons]
    summary=dict(generated_utc=datetime.now(timezone.utc).isoformat(), notebook_executed=False,
        notebook_modified=False, source_hashes=before, exported_images=images,
        saved_notebook_sizes=sizes.tolist(), beta_min=float(beta.min()), beta_max=float(beta.max()),
        beta_count=len(beta), beta_step=float(np.median(np.diff(beta))), beta_contains_zero=bool(np.any(beta==0)),
        L_MIN=3, BETA_STEP=12, examples_beta=beta[np.unique(np.r_[np.arange(0,len(beta),12),len(beta)-1])].tolist(),
        fitted_quantity="Delta S2 = S2(A)-S2(B)", logarithm="natural",
        saved_fit_model="alpha*ln(L)+constant; equal weights; no inverse-size correction; all 59 beta values fitted",
        region_A="(L/3) x L cells, both sublattices", region_B="(L/3) x (2L/3) cells, both sublattices",
        current_generator="Ground state of H0 + mu*diag(-1,+1), mu=1e-5; even B indices multiplied by exp(-2*beta), followed by thin QR.",
        generator_provenance_limit="Current test.jl active L_set is [6,12,18,24,30,36], unlike saved CSV sizes [3,6,...,21]. Its present source is not a saved execution manifest for these CSV files.",
        spin_normalization="-logdet(C^2+(I-C)^2) is one species. No overall factor two is applied in this notebook/generator; it must not be conflated with the different spin-doubled Ref notebook.",
        relation_to_current_D="For orientation-symmetric uniform_A, SA=SC and D=SA-SB. Compare this notebook with D, not 2D, at matched states and fitting windows. Two identical independent species would instead have 2D.",
        saved_alpha_plot=dict(figsize_inches=[7,4], constrained_layout=True, color="#2166AC",
            marker="o", line="solid", markersize=4, x_axis="beta linear", y_axis="alpha linear",
            horizontal_zero_line="gray0.5, linewidth0.8", grid_alpha=.2, legend="none",
            title="Logarithmic fit: L = 3–21 (7 sizes)", error_bars=False,
            inline_figure_dpi=120, saved_file_dpi=200),
        scaling_example_plot=dict(x_axis="ln L, already transformed", legend=["Data","Fit"],
            data_color="#2166AC", fit_color="#D47719", independent_y_scales=True),
        max_delta_identity_error=identity_error, max_saved_fit_reproduction_error=fit_difference,
        normalization_pairs=len(comparisons), saved_over_current_D_min=min(ratios), saved_over_current_D_max=max(ratios),
        max_difference_same_single_species=max(abs(r["saved_minus_current_D"]) for r in comparisons),
        max_difference_if_doubled=max(abs(r["saved_minus_twice_current_D"]) for r in comparisons),
        no_solver_changed=True)
    after={str(path):sha(path) for path in SOURCES}
    assert before==after, "A read-only source changed during inspection"
    (HERE / "inspection_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k:summary[k] for k in ("saved_notebook_sizes","beta_count","max_saved_fit_reproduction_error",
        "normalization_pairs","saved_over_current_D_min","saved_over_current_D_max",
        "max_difference_same_single_species","max_difference_if_doubled")},ensure_ascii=False))


if __name__=="__main__":
    main()
