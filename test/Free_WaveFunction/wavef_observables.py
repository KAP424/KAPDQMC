"""Observables and lattice plots for a number-conserving Gaussian fermion state.

Throughout, P[i,j] = <c_j^dagger c_i>, matching test/test.jl.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.colors import Normalize, TwoSlopeNorm, SymLogNorm
from matplotlib.lines import Line2D


OPERATORS = {
    "N": np.eye(2, dtype=complex),
    "Sx": np.array([[0, 1], [1, 0]], dtype=complex) / 2,
    "Sy": np.array([[0, -1j], [1j, 0]], dtype=complex) / 2,
    "Sz": np.diag([1, -1]).astype(complex) / 2,
    "DW": np.diag([1, -1]).astype(complex),
}


def real_checked(values, tol=1e-10):
    """Reject a complex result instead of silently hiding a convention error."""
    values = np.asarray(values)
    if np.max(np.abs(values.imag), initial=0.0) > tol:
        raise ValueError("Expected a real observable; check operator order/convention.")
    return values.real


def local_mean(P, indices, operator):
    return np.trace(operator @ P[np.ix_(indices, indices)])


def connected(P, indices_a, operator_a, indices_b, operator_b):
    """<O_A O_B>_c = Tr[A (I-P)_AB B P_BA], including overlapping support.

    O_A=c_A^dagger A c_A. Order matters for noncommuting operators.
    For the same Hermitian operator in separate cells the result is real.
    """
    ia, ib = np.asarray(indices_a), np.asarray(indices_b)
    pab = P[np.ix_(ia, ib)]
    pba = P[np.ix_(ib, ia)]
    delta_ab = (ia[:, None] == ib[None, :]).astype(float)
    return np.trace(operator_a @ (delta_ab - pab) @ operator_b @ pba)


def density_connected(P):
    return np.diag(real_checked(np.diag(P))) - np.abs(P) ** 2


def cell_observables(P):
    cells = np.arange(P.shape[0]).reshape(-1, 2)
    means, variances = {}, {}
    for name, op in OPERATORS.items():
        means[name] = real_checked([local_mean(P, ids, op) for ids in cells])
        variances[name] = real_checked([connected(P, ids, op, ids, op) for ids in cells])
    return means, variances


def reference_correlations(P, reference_cell):
    cells = np.arange(P.shape[0]).reshape(-1, 2)
    ref = cells[reference_cell]
    return {
        name: real_checked([connected(P, ids, op, ref, op) for ids in cells])
        for name, op in OPERATORS.items()
    }


def lattice_geometry(metadata, reference_xy):
    """Center a PBC parallelogram at a one-based (x,y) reference cell.

    Wrapped primitive coordinates define a fundamental domain, not a Euclidean
    minimum-image distance; each original cell occurs exactly once.
    """
    lx, ly = metadata["Lx"], metadata["Ly"]
    ref = np.asarray(reference_xy, dtype=int) - 1
    if ref.shape != (2,) or not (0 <= ref[0] < lx and 0 <= ref[1] < ly):
        raise ValueError("Reference cell is outside the lattice.")
    cell_xy = np.array([(x, y) for y in range(ly) for x in range(lx)])
    shape = np.array([lx, ly])
    displacements = (cell_xy - ref + shape // 2) % shape - shape // 2
    basis = np.array([metadata["a1"], metadata["a2"]])
    centers = displacements @ basis
    delta = np.array(metadata["delta_AB"])
    positions = (centers[:, None, :] + np.array([-delta / 2, delta / 2])).reshape(-1, 2)
    bond_pairs = []
    for x, y in cell_xy:
        i = 2 * (x + lx * y)
        for bx, by in ((x, y), ((x + 1) % lx, y), (x, (y - 1) % ly)):
            bond_pairs.append((i, 2 * (bx + lx * by) + 1))
    bond_pairs = np.array(bond_pairs, dtype=int)
    return {
        "cell_xy": cell_xy, "displacements": displacements, "basis": basis,
        "centers": centers, "positions": positions, "bond_pairs": bond_pairs,
        "reference_cell": int(ref[0] + lx * ref[1]),
    }


def signed_norm(values, *, mode="linear", limit=None, linthresh=1e-5):
    values = np.asarray(values)
    vmax = max(float(np.max(np.abs(values), initial=0)), 1e-12) if limit is None else limit
    if mode == "symlog":
        return SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax, base=10)
    if mode != "linear":
        raise ValueError("mode must be linear or symlog")
    return TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)


def decorate_axis(ax):
    ax.set_aspect("equal")
    ax.set_xlabel(r"$r_x / a_{\mathrm{NN}}$")
    ax.set_ylabel(r"$r_y / a_{\mathrm{NN}}$")
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(0.04)


def range_caption(ax, values):
    values = np.asarray(values)
    ax.text(0, 1.015, f"min {values.min():.6g}   max {values.max():.6g}",
            transform=ax.transAxes, fontsize=9, color="0.35")


def cell_map(ax, values, geometry, title, *, norm=None, reference=False,
             cmap="RdBu_r", mask_reference=False):
    values = np.asarray(values)
    if len(values) != len(geometry["centers"]):
        raise ValueError("One value per cell is required.")
    a1, a2 = geometry["basis"]
    corners = np.array([-a1-a2, a1-a2, a1+a2, -a1+a2]) * 0.48
    polygons = geometry["centers"][:, None, :] + corners
    data = np.ma.array(values, mask=np.zeros(len(values), dtype=bool))
    if mask_reference:
        data.mask[geometry["reference_cell"]] = True
    palette = plt.get_cmap(cmap).copy()
    palette.set_bad("#dddada")
    artist = PolyCollection(polygons, array=data, cmap=palette,
                            norm=norm or signed_norm(values), edgecolors="white", linewidths=0.3)
    ax.add_collection(artist)
    ax.autoscale_view()
    ax.set_title(title, pad=28)
    range_caption(ax, values[~data.mask])
    if reference:
        ax.scatter([0], [0], facecolors="none", edgecolors="black", s=65, linewidths=1.2)
    decorate_axis(ax)
    ax.figure.colorbar(artist, ax=ax, fraction=0.04, pad=0.025)
    return artist


def visible_bonds(geometry):
    segments = geometry["positions"][geometry["bond_pairs"]]
    visible = np.isclose(np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1), 1.0)
    return segments[visible], visible


def site_map(ax, values, geometry, title, *, norm=None, reference_site=None, cmap="RdBu_r"):
    values = np.asarray(values)
    positions = geometry["positions"]
    segments, _ = visible_bonds(geometry)
    ax.add_collection(LineCollection(segments, colors="#c7c7c7", linewidths=0.7, zorder=0))
    norm = norm or signed_norm(values)
    for sub, marker in ((0, "o"), (1, "^")):
        ids = np.arange(sub, len(values), 2)
        artist = ax.scatter(*positions[ids].T, c=values[ids], cmap=cmap, norm=norm,
                            marker=marker, s=34, edgecolors="#444444", linewidths=0.3)
    handles = [Line2D([], [], color="0.4", marker=m, linestyle="none", label=label)
               for m, label in (("o", "A"), ("^", "B"))]
    ax.legend(handles=handles, loc="upper right", frameon=False, ncol=2)
    if reference_site is not None:
        ax.scatter(*positions[reference_site], facecolors="none", edgecolors="black",
                   s=140, linewidths=1.3)
    ax.set_title(title, pad=28)
    range_caption(ax, values)
    decorate_axis(ax)
    ax.figure.colorbar(artist, ax=ax, fraction=0.04, pad=0.025)
    return artist


def save_figure(fig, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
