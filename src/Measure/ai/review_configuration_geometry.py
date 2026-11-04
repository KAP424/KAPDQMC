"""Recount existing configuration censuses; no new Hamiltonian calculations.

Run from any directory with Python's standard library. All outputs are separate
from the source censuses and the locked equilibrium/extraction analyses.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def move(mask, m, dx=0, dy=0, mirror=False):
    """Zero-based cells; A/B labels stay fixed under translations and R_a."""
    value = 0
    for y in range(m):
        for x in range(m):
            xx, yy = (-y, -x) if mirror else (x, y)
            xx, yy = (xx + dx) % m, (yy + dy) % m
            for sub in (0, 1):
                if (mask >> (sub + 2 * (x + m * y))) & 1:
                    value |= 1 << (sub + 2 * (xx + m * yy))
    return value


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "analysis_geometry")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = [HERE / "data/configuration_census.csv", HERE / "data/configuration_census_sizes.csv"]
    rows, receipts = [], []
    for path in sources:
        data = path.read_bytes()
        receipts.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(data).hexdigest()))
        for line, row in enumerate(csv.DictReader(data.decode("utf-8-sig").splitlines()), 2):
            for key in ("m", "L", "mask", "doubles", "sublattice_imbalance"):
                row[key] = int(row[key])
            row["special"] = row["special"] == "true"
            row["ground"] = abs(float(row["excess_energy"])) < 1e-8
            row["source_file"], row["source_line"] = path.name, line
            expected_bits = "".join(str((row["mask"] >> j) & 1) for j in range(2 * row["m"] ** 2))
            assert row["bitstring"] == expected_bits and expected_bits.count("1") == row["m"] ** 2
            rows.append(row)

    cache = {}
    for row in rows:
        key = row["m"], row["mask"]
        if key not in cache:
            m, mask = key
            translations = [(dx, dy) for dy in range(m) for dx in range(m)
                            if move(mask, m, dx, dy) == mask]
            px = next(dx for dx in range(1, m + 1) if move(mask, m, dx, 0) == mask)
            py = next(dy for dy in range(1, m + 1) if move(mask, m, 0, dy) == mask)
            mirrors = [a for a in range(m) if move(mask, m, a, a, True) == mask]
            cache[key] = dict(period_x=px, period_y=py,
                              translation_stabilizer_count=len(translations),
                              extra_translation=len(translations) > 1,
                              mirror_R0=0 in mirrors, mirror_Ra_any=bool(mirrors),
                              mirror_Ra_offsets=",".join(map(str, mirrors)))
        row.update(cache[key])
        row["AB_equal"] = row["sublattice_imbalance"] == 0
        row["abs_imbalance"] = abs(row["sublattice_imbalance"])
        row["n_A"] = (row["m"] ** 2 + row["sublattice_imbalance"]) // 2
        row["n_B"] = (row["m"] ** 2 - row["sublattice_imbalance"]) // 2
        row["enumeration_scope"] = "special_only" if row["L"] in (27, 36) else "all_half_filled"

    counts = defaultdict(lambda: [0, 0])
    features = ("AB_equal", "abs_imbalance", "extra_translation", "translation_stabilizer_count",
                "period_x", "period_y", "mirror_R0", "mirror_Ra_any")
    for row in rows:
        for family in ("all", "special" if row["special"] else "general"):
            for feature in features:
                key = (row["m"], row["L"], row["enumeration_scope"], family, feature, str(row[feature]))
                counts[key][0] += 1
                counts[key][1] += row["ground"]
    grouped = [dict(zip(("m", "L", "enumeration_scope", "family", "feature", "value"), key),
                    total=count[0], ground_count=count[1], blocked_count=count[0] - count[1])
               for key, count in sorted(counts.items())]
    write_csv(args.output / "geometry_counts.csv", grouped)

    selected = {(1, 3, 1), (2, 6, 15), (2, 6, 27), (2, 12, 27)}
    selected.update((3, size, 87402) for size in (9, 18, 27, 36))
    selected.update((3, size, mask) for size in (9, 18) for mask in (43258, 44203))
    catalogue_path = HERE / "data/patterns.csv"
    catalogue_data = catalogue_path.read_bytes()
    catalogue = list(csv.DictReader(catalogue_data.decode("utf-8-sig").splitlines()))
    for name in ("one_B_2", "checker_AB", "diagonal_B_3"):
        pattern = next(p for p in catalogue if p["pattern"] == name)
        mask = sum(int(bit) << j for j, bit in enumerate(pattern["occupied_bits"]))
        m = int(pattern["m"])
        selected.add((m, 6 if m == 2 else 18, mask))
    examples = []
    for row in rows:
        if (row["m"], row["L"], row["mask"]) not in selected:
            continue
        item = {key: row[key] for key in ("m", "L", "mask", "bitstring", "special", "n_A", "n_B",
                "doubles", "excess_energy", "ground", "period_x", "period_y", "translation_stabilizer_count",
                "mirror_R0", "mirror_Ra_any", "mirror_Ra_offsets", "source_file", "source_line")}
        bits, m = row["bitstring"], row["m"]
        names = {"10": "A", "01": "B", "11": "AB", "00": "empty"}
        item["cell_rows_y"] = " / ".join(" ".join(names[bits[2*(x+m*y):2*(x+m*y)+2]]
                                                  for x in range(m)) for y in range(m))
        examples.append(item)
    write_csv(args.output / "geometry_counterexamples.csv", examples)

    metadata = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), sources=receipts,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        catalogue_source=dict(path=str(catalogue_path.resolve()), sha256=hashlib.sha256(catalogue_data).hexdigest()),
        source_rows=len(rows), unique_pattern_geometry_entries=len(cache),
        action="Only transformations/counting of existing exhaustive CSV rows; no new Hamiltonian calculations, fits, simulations, or changes to locked inputs.",
        ground="abs(excess_energy)<1e-8; not negative_holes alone",
        bit_order="A(x,y),B(x,y), x fastest; source bitstring lists low bit first; diagram rows fix y",
        period="period_x/y are least positive translation offsets preserving all A/B occupations; extra_translation means any nonzero torus translation preserving pattern",
        mirror="R_a(x,y,s)=(-y+a,-x+a,s) modulo m, a=0..m-1; no A/B swap. This graph reflection maps hopping offset set {(0,0),(1,0),(0,-1)} into itself; R_a squared is identity.",
        coordinate_origin="Transforms use zero-based x,y=0..m-1. For 1-based labels, R0 is x_new=1+mod(-(y-1),m), y_new=1+mod(-(x-1),m). Display rows enumerate y in ascending order.",
        caveats=["m is declared supercell size and can exceed a primitive period; parity of m alone is not intrinsic to a pattern.",
                 "At m3 L27/L36 only special patterns were enumerated; counts labelled all there still refer only to that enumerated subset.",
                 "mirror_Ra_any checks one specified reflection family, not all honeycomb point-group symmetries.",
                 "Counts are exact for the recorded finite sizes, not asymptotic classification theorems.",
                 "Mirror/count/period feature agreement does not claim identical complete space groups."])
    (args.output / "geometry_provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    assert all(hashlib.sha256(Path(r["path"]).read_bytes()).hexdigest() == r["sha256"] for r in receipts)
    print(json.dumps(dict(source_rows=len(rows), grouped_rows=len(grouped), example_rows=len(examples),
                          output=str(args.output.resolve())), indent=2))


if __name__ == "__main__":
    main()
