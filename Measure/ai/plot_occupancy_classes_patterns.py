"""Render the frozen six-pattern catalogue and summarize its reachability CSV.

No simulation or fitting is performed. The Julia catalogue is read as literal
data and checked against every reachability record before publishing the figure.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
PATTERNS = HERE / "occupancy_classes_patterns.jl"
REACHABILITY = HERE / "data/occupancy_classes_reachability.csv"
KERNEL = HERE / "fermion_experiment.jl"
CHECK_SCRIPT = HERE / "check_occupancy_classes_reachability.jl"
OUT = HERE / "figures_occupancy_classes"
COLORS = ("#236192", "#B45C23")
CLASSES = {
    1: "Class 1  |  One fermion in every cell",
    2: "Class 2  |  Only empty and doubly occupied cells",
    3: "Class 3  |  Empty, single and double cells coexist",
}
SIZES = list(range(24, 145, 12)) + [180, 216]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_inputs():
    entries = re.findall(
        r'\("([^"]+)",\s*"([^"]+)",\s*(\d+),\s*"([01]+)",\s*(\d+)\)',
        PATTERNS.read_text(encoding="utf-8"),
    )
    assert len(entries) == 6
    patterns = []
    for name, family, m_text, bits, cls_text in entries:
        m, cls = int(m_text), int(cls_text)
        assert len(bits) == 2*m*m and bits.count("1") == m*m
        n = [int(bits[2*i]) + int(bits[2*i+1]) for i in range(m*m)]
        counts = [n.count(j) for j in range(3)]
        assert counts[0] == counts[2]
        assert (cls == 1) == (counts[1] == m*m)
        assert (cls == 2) == (counts[1] == 0)
        assert (cls == 3) == all(value > 0 for value in counts)
        patterns.append(dict(pattern=name, family=family, m=m, class_id=cls,
                             occupied_bits=bits, cell_counts=counts))
    assert all(sum(p["class_id"] == cls for p in patterns) == 2 for cls in CLASSES)
    with REACHABILITY.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {(p["pattern"], L) for p in patterns for L in SIZES}
    actual = {(r["pattern"], int(r["L"])) for r in rows}
    assert actual == expected and len(rows) == len(expected)
    definitions = {p["pattern"]: p for p in patterns}
    for row in rows:
        p = definitions[row["pattern"]]
        assert row["occupied_bits"] == p["occupied_bits"]
        assert int(row["class_id"]) == p["class_id"] and int(row["m"]) == p["m"]
        assert row["pattern_sha256"] == sha256(PATTERNS)
        assert row["kernel_sha256"] == sha256(KERNEL)
        assert [int(row[k]) for k in ("empty_cells", "single_cells", "double_cells")] == p["cell_counts"]
        assert (row["ground_reachable"] == "true") == (
            abs(float(row["excess_energy"])) < float(row["ground_tol"])
        )
    return patterns, rows


def draw_panel(ax, p):
    m, bits = p["m"], p["occupied_bits"]
    for y in range(m):
        for x in range(m):
            ax.add_patch(Rectangle((x-.46, y-.40), .92, .80,
                                   facecolor="#F6F8FA", edgecolor="#D9DFE5",
                                   linewidth=.75, zorder=0))
            for sub in (0, 1):
                occupied = bits[2*(x+m*y)+sub] == "1"
                ax.scatter(x + (-.17 if sub == 0 else .17), y,
                           s=100, marker="o" if sub == 0 else "s",
                           facecolor=COLORS[sub] if occupied else "white",
                           edgecolor=COLORS[sub], linewidth=1.5, zorder=3)
    ax.set(xlim=(-.6, m-.4), ylim=(-.55, m-.45), aspect="equal")
    ax.set_xticks(range(m), [str(i+1) for i in range(m)])
    ax.set_yticks(range(m), [str(i+1) for i in range(m)])
    ax.set_xlabel("x", fontsize=11, labelpad=3)
    ax.set_ylabel("y", fontsize=11, rotation=0, labelpad=8)
    ax.tick_params(length=0, labelsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(p["pattern"], fontsize=14, fontweight="bold", pad=31)
    n0, n1, n2 = p["cell_counts"]
    ax.text(.5, 1.035, f"m = {m}     |     (n0, n1, n2) = ({n0}, {n1}, {n2})",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=11)


def main():
    patterns, rows = read_inputs()
    OUT.mkdir(exist_ok=True)
    source_hashes = {str(path.relative_to(HERE)): sha256(path) for path in
                     (PATTERNS, REACHABILITY, KERNEL, CHECK_SCRIPT)}
    pattern_summaries = []
    for p in patterns:
        group = [row for row in rows if row["pattern"] == p["pattern"]]
        pattern_summaries.append({
            **p,
            "checked_sizes": sorted(int(row["L"]) for row in group),
            "ground_reachable_rows": sum(row["ground_reachable"] == "true" for row in group),
            "max_abs_excess_energy": max(abs(float(row["excess_energy"])) for row in group),
            "max_reconstruction_error": max(float(row["reconstruction_error"]) for row in group),
            "min_pivot_sv": min(float(row["min_pivot_sv"]) for row in group),
        })
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Premeasurement ground-state reachability at listed finite sizes only. "
                 "No claim of entropy scaling or reachability at all sizes.",
        "rows": len(rows),
        "expected_rows": 6*len(SIZES),
        "complete": True,
        "patterns_count": len(patterns),
        "sizes": SIZES,
        "L_min": min(SIZES),
        "L_max": max(SIZES),
        "all_ground_reachable": all(row["ground_reachable"] == "true" for row in rows),
        "max_abs_excess_energy": max(abs(float(row["excess_energy"])) for row in rows),
        "max_reconstruction_error": max(float(row["reconstruction_error"]) for row in rows),
        "min_pivot_sv": min(float(row["min_pivot_sv"]) for row in rows),
        "rank_tolerances": sorted({float(row["rank_tol"]) for row in rows}),
        "ground_tolerances": sorted({float(row["ground_tol"]) for row in rows}),
        "source_sha256": source_hashes,
        "summary_script_sha256": sha256(Path(__file__)),
        "patterns": pattern_summaries,
    }
    summary_path = HERE / "data/occupancy_classes_reachability_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "text.color": "#222B34", "axes.labelcolor": "#222B34"})
    fig = plt.figure(figsize=(11.2, 15))
    fig.suptitle("Six periodic occupation configurations", y=.985,
                 fontsize=19, fontweight="bold")
    fig.text(.5, .956, "Half filling: one fermion per cell on average",
             ha="center", fontsize=12)
    legend = []
    for sub, label in enumerate(("A", "B")):
        for occupied in (True, False):
            legend.append(Line2D([], [], marker="o" if sub == 0 else "s",
                                 color="none", markersize=8,
                                 markerfacecolor=COLORS[sub] if occupied else "white",
                                 markeredgecolor=COLORS[sub], markeredgewidth=1.5,
                                 label=f"{label} {'occupied' if occupied else 'empty'}"))
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5,.937),
               ncol=4, frameon=False, fontsize=12)
    for row_index, cls in enumerate(CLASSES):
        bottom = .66 - .275*row_index
        fig.text(.075, bottom+.230, CLASSES[cls], fontsize=14, fontweight="bold",
                 ha="left", va="center")
        for column, p in enumerate(p for p in patterns if p["class_id"] == cls):
            ax = fig.add_axes((.08 + .46*column, bottom, .38, .16))
            draw_panel(ax, p)
    fig.text(.5, .018,
             "n0 / n1 / n2: numbers of empty / single / double cells in one periodic tile.\n"
             "Cell-index diagram: x right, y up; grey box = one A/B cell. Not the real-space honeycomb geometry.",
             ha="center", va="bottom", fontsize=10.5, linespacing=1.6)
    files = []
    for suffix in ("png", "pdf"):
        path = OUT / f"pattern_catalogue.{suffix}"
        fig.savefig(path, dpi=200, facecolor="white")
        files.append({"path": path.name, "sha256": sha256(path)})
    plt.close(fig)
    manifest = {
        "source_sha256": source_hashes,
        "script_sha256": sha256(Path(__file__)),
        "reachability_summary_sha256": sha256(summary_path),
        "files": files,
        "panels": patterns,
        "bit_order": "A(x,y),B(x,y); x fastest; x and y are 1-based in the diagram",
        "axes": "x rightward, y upward",
        "encoding": "A blue circle, B orange square; filled=occupied, hollow=empty",
        "new_simulations_or_fits": False,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"png": str(OUT/"pattern_catalogue.png"), "rows": len(rows),
                      "all_ground_reachable": summary["all_ground_reachable"],
                      "max_abs_excess_energy": summary["max_abs_excess_energy"],
                      "max_reconstruction_error": summary["max_reconstruction_error"]}))


if __name__ == "__main__":
    main()
