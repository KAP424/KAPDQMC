"""Draw the exact saved configurations referenced by SCALING_GUIDE.md.

No simulations or fits are run. x increases rightward, y upward; the source
bit order is A(1,1), B(1,1), A(2,1), B(2,1), ... (x varies first).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


HERE = Path(__file__).resolve().parent
OUT = HERE / "figures_scaling_guide"
GROUPS = [
    ("special_patterns", "Eight special configurations: ground-state accessible at sampled sizes",
     ["uniform_A", "uniform_B", "one_B_2", "one_B_3", "two_B_3",
      "diagonal_B_3", "special_matched_a", "special_matched_b"], 4),
    ("general_patterns", "Four general configurations: ground-state accessible at sampled sizes",
     ["full_empty_cluster3", "random_half_3", "general_matched_a", "general_matched_b"], 2),
    ("obstructed_patterns", "Five ground-state-obstructed controls at sampled sizes",
     ["stripe_AB_x", "stripe_AB_y", "checker_AB", "full_empty_x", "full_empty_checker"], 3),
]
COLORS = ("#236192", "#B45C23")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def draw_panel(ax, name, definition, ground_sizes, obstructed):
    m = int(definition["m"])
    bits = [int(value) for value in definition["occupied_bits"]]
    assert len(bits) == 2*m*m and sum(bits) == m*m, name
    cells = [bits[2*i:2*i+2] for i in range(m*m)]
    empty = sum(sum(cell) == 0 for cell in cells)
    double = sum(sum(cell) == 2 for cell in cells)
    assert empty == int(definition["empty_cells"]), name
    assert double == int(definition["double_cells"]), name
    assert (empty == double == 0) == (definition["family"] == "special"), name

    for y in range(m):
        for x in range(m):
            ax.add_patch(Rectangle((x-.46, y-.40), .92, .80,
                                   facecolor="#F6F8FA", edgecolor="#D9DFE5", lw=.7, zorder=0))
            for sub in (0, 1):
                occupied = bits[2*(x+m*y)+sub]
                ax.scatter(x+(-.17 if sub == 0 else .17), y,
                           s=100, marker="o" if sub == 0 else "s",
                           facecolor=COLORS[sub] if occupied else "white",
                           edgecolor=COLORS[sub], linewidth=1.5, zorder=3)
    ax.set(xlim=(-.6, m-.4), ylim=(-.55, m-.45), aspect="equal")
    ax.set_xticks(range(m), [str(i+1) for i in range(m)])
    ax.set_yticks(range(m), [str(i+1) for i in range(m)])
    ax.set_xlabel("x", fontsize=10, labelpad=2)
    ax.set_ylabel("y", fontsize=10, rotation=0, labelpad=7)
    ax.tick_params(length=0, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)

    na, nb = sum(bits[::2]), sum(bits[1::2])
    metadata = f"{m} x {m} cells | N_A/N_B = {na}/{nb} | empty/double = {empty}/{double}"
    if obstructed:
        status = "Projection: obstructed at checked sizes"
    else:
        status = "Projection: GS accessible at checked sizes"
    ax.set_title(name, fontsize=13, fontweight="bold", pad=39)
    ax.text(.5, 1.085, metadata, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=8.5)
    ax.text(.5, 1.015, status, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=9)
    return dict(pattern=name, m=m, occupied_bits=definition["occupied_bits"],
                A_particles=na, B_particles=nb, empty_cells=empty, double_cells=double,
                projection_classification="obstructed" if obstructed else "ground_state_accessible",
                checked_ground_sizes=ground_sizes)


def main():
    sources = [HERE/"data/patterns.csv", HERE/"data/arrangement_selection.csv",
               HERE/"data/equilibrium_validation.json"]
    definitions = {row["pattern"]:row for path in sources[:2] for row in read_csv(path)}
    gs = json.loads(sources[2].read_text())["pattern_ground_sizes"]
    assert len({name for _, _, names, _ in GROUPS for name in names}) == 17
    assert set(name for _, _, names, _ in GROUPS[:2] for name in names) == {name for name,sizes in gs.items() if sizes}
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans", "pdf.fonttype":42})
    legend = [
        Line2D([], [], marker="o", color="none", markerfacecolor=COLORS[0],
               markeredgecolor=COLORS[0], markersize=8, label="A occupied"),
        Line2D([], [], marker="o", color="none", markerfacecolor="white",
               markeredgecolor=COLORS[0], markersize=8, label="A empty"),
        Line2D([], [], marker="s", color="none", markerfacecolor=COLORS[1],
               markeredgecolor=COLORS[1], markersize=8, label="B occupied"),
        Line2D([], [], marker="s", color="none", markerfacecolor="white",
               markeredgecolor=COLORS[1], markersize=8, label="B empty"),
    ]
    records = []
    for stem, title, names, columns in GROUPS:
        obstructed = stem == "obstructed_patterns"
        rows = math.ceil(len(names)/columns)
        fig, axes = plt.subplots(rows, columns, figsize=(4.1*columns, 4.2*rows+.6), squeeze=False)
        fig.subplots_adjust(left=.045, right=.985, top=.80, bottom=.12, wspace=.30, hspace=.73)
        panels = []
        for ax, name in zip(axes.flat, names):
            assert bool(gs[name]) != obstructed, name
            panels.append(draw_panel(ax, name, definitions[name], gs[name], obstructed))
        for ax in list(axes.flat)[len(names):]:
            ax.axis("off")
        fig.suptitle(title, fontsize=16, y=.985)
        fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5,.95),
                   ncol=4, frameon=False, fontsize=11)
        fig.text(.5,.018,
                 "Cell-index diagram: x right, y up; grey box = one A/B cell. Not the real-space honeycomb geometry.\n"
                 "Projection labels refer to premeasurement ground-state reachability at checked sizes, not to scaling coefficients.",
                 ha="center", va="bottom", fontsize=10)
        files = []
        for extension in ("png", "pdf"):
            path = OUT/f"{stem}.{extension}"
            fig.savefig(path, dpi=180, facecolor="white")
            files.append(dict(path=path.name, sha256=sha256(path)))
        plt.close(fig)
        records.append(dict(group=stem, panels=panels, files=files))
    manifest = dict(source_files=[dict(path=str(p.relative_to(HERE)), sha256=sha256(p)) for p in sources],
                    script_sha256=sha256(Path(__file__)), groups=records,
                    bit_order="A(1,1),B(1,1),A(2,1),B(2,1),...; x varies first",
                    diagram_axes="x rightward; y upward", no_new_simulations_or_fits=True,
                    classification_scope="Premeasurement ground-state reachability at checked sizes only; no scaling support labels.")
    (OUT/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Rendered 17 exact configurations in three PNG/PDF pairs.")


if __name__ == "__main__":
    main()
