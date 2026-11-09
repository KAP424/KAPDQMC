"""Independent four-orbital check of alternating-state overlap boundary layers.

This uses primitive-cell momentum pairs, not the entropy experiment backend.
It tests a local projection mechanism only; no region entropy or fitted corner
coefficient is computed.  Exact zero-overlap momenta use explicit supported
orbitals instead of inverting their singular negative-energy overlap matrix.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np


BASE = Path(__file__).resolve().parent
OUT = BASE / "analysis_second_class_extra"
THETAS = (4.0, 7.0)
SCALED_DELTAS = (0.1, 1.0, 10.0)
SIZES = (144, 180)
P0 = np.vstack((np.eye(2), np.eye(2))) / math.sqrt(2)
PATTERNS = {
    "pure02_alternating_x": dict(width_prefactor=3.0, sigma_slope=1.0 / 6),
    "pure02_alternating_checker": dict(width_prefactor=2.0, sigma_slope=0.5),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bloch_pair(pattern, delta):
    """h(q)=[[0,f(q)],[conj(f(q)),0]], f=1+exp(i qx)+exp(-i qy)."""
    if pattern == "pure02_alternating_x":
        momenta = ((0.0, delta), (math.pi, delta))
        fs = (2 + np.exp(-1j * delta), np.exp(-1j * delta))
    else:
        momenta = ((math.pi + delta, 0.0), (delta, math.pi))
        fs = (2 - np.exp(1j * delta), np.exp(1j * delta))
    h = np.zeros((4, 4), complex)
    for j, f in enumerate(fs):
        h[2 * j, 2 * j + 1] = f
        h[2 * j + 1, 2 * j] = f.conjugate()
    return h, momenta


def exact_zero_support(pattern, theta):
    """Finite-time and infinite-time orbitals at the EXACT blocked momentum."""
    minus = np.array([-1.0, 1.0]) / math.sqrt(2)
    plus = np.array([1.0, 1.0]) / math.sqrt(2)
    q = np.zeros((4, 2), complex)
    if pattern == "pure02_alternating_x":
        # Energies in the two primitive momentum blocks are +/-3 and +/-1.
        # Each sigma_x branch contains one initial particle.  No evolution
        # can transfer that positive-branch particle to the negative branch.
        r = 0.0 if math.isinf(theta) else math.exp(-2 * theta)
        q[:, 0] = np.concatenate((minus, r * minus)) / math.sqrt(1 + r * r)
        q[:, 1] = np.concatenate((r * plus, plus)) / math.sqrt(1 + r * r)
    else:
        # Both primitive blocks have h=sigma_x.  Evolution changes the two
        # occupied basis columns internally, leaving their projector fixed.
        q[:, 0] = np.concatenate((minus, minus)) / math.sqrt(2)
        q[:, 1] = np.concatenate((plus, plus)) / math.sqrt(2)
    return q


def observe(pattern, delta, theta, point_type, size=None, scaled_delta=None):
    h, momenta = bloch_pair(pattern, delta)
    e, v = np.linalg.eigh(h)
    assert e[1] < 0 < e[2], "These local diagnostics must remain gapped."
    a = v[:, :2].conj().T @ P0
    b = v[:, 2:].conj().T @ P0
    singular = np.linalg.svd(a, compute_uv=False)
    if delta == 0:
        q = exact_zero_support(pattern, theta)
        method = "exact_zero_overlap_supported_orbitals"
        graph_norm = None
    else:
        assert not math.isinf(theta)
        x0 = np.linalg.solve(a.T, b.T).T
        x = x0 * np.exp(-theta * (e[2:, None] - e[None, :2]))
        frame = np.linalg.qr(np.vstack((np.eye(2), x)))[0]
        q = v @ frame
        graph_norm = float(np.linalg.svd(x, compute_uv=False)[0])
        method = "invertible_negative_overlap_graph"
    c = q @ q.conj().T
    n_positive = float(np.sum(np.abs(v[:, 2:].conj().T @ q) ** 2))
    width = 0.0 if math.isinf(theta) else PATTERNS[pattern]["width_prefactor"] * math.exp(-2 * theta)
    # At delta=0 support is exactly blocked, even after theta->infinity.
    predicted = 1.0 if delta == 0 else 1 / (1 + (delta / width) ** 2)
    orthogonality_error = float(np.linalg.norm(q.conj().T @ q - np.eye(2)))
    projector_error = float(np.linalg.norm(c @ c - c))
    energy = float(np.trace(h @ c).real)
    row = dict(pattern=pattern, point_type=point_type, L=size,
        theta="Inf" if math.isinf(theta) else theta, delta=delta,
        delta_over_width=scaled_delta if scaled_delta is not None else (delta / width if width else None),
        q1_x=momenta[0][0], q1_y=momenta[0][1], q2_x=momenta[1][0], q2_y=momenta[1][1],
        method=method, negative_overlap_rank=int(np.count_nonzero(singular > 1e-10)),
        negative_dimension=2, sigma_min=float(singular[-1]),
        sigma_min_over_delta=float(singular[-1] / delta) if delta else None,
        sigma_slope_asymptotic=PATTERNS[pattern]["sigma_slope"],
        positive_negative_gap=float(e[2] - e[1]),
        width_asymptotic=width, graph_norm=graph_norm,
        n_positive=n_positive, n_positive_asymptotic=predicted,
        n_positive_asymptotic_abs_error=abs(n_positive - predicted),
        energy=energy, ground_energy=float(e[:2].sum()),
        excess_energy=energy - float(e[:2].sum()),
        orthogonality_error=orthogonality_error, projector_error=projector_error)
    return row


def main():
    rows = []
    for pattern, parameters in PATTERNS.items():
        for theta in THETAS:
            width = parameters["width_prefactor"] * math.exp(-2 * theta)
            for ratio in SCALED_DELTAS:
                rows.append(observe(pattern, ratio * width, theta, "fixed_delta_over_width", scaled_delta=ratio))
            for size in SIZES:
                rows.append(observe(pattern, 2 * math.pi / size, theta, "nearest_nonzero_grid_momentum", size=size))
            rows.append(observe(pattern, 0.0, theta, "exact_blocked_momentum"))
        rows.append(observe(pattern, 0.0, math.inf, "exact_blocked_momentum_limit"))
    local = [r for r in rows if r["point_type"] == "fixed_delta_over_width"]
    zeros = [r for r in rows if r["delta"] == 0]
    error = max(r["n_positive_asymptotic_abs_error"] for r in local)
    sigma_error = max(abs(r["sigma_min_over_delta"] / r["sigma_slope_asymptotic"] - 1) for r in local)
    orth_error = max(r["orthogonality_error"] for r in rows)
    projector_error = max(r["projector_error"] for r in rows)
    zero_error = max(abs(r["n_positive"] - 1) for r in zeros)
    assert len(rows) == 26 and len(local) == 12
    assert error < 5e-6 and sigma_error < 1e-4
    assert orth_error < 1e-12 and projector_error < 1e-12 and zero_error < 1e-12
    assert all(r["negative_overlap_rank"] == (1 if r["delta"] == 0 else 2) for r in rows)
    OUT.mkdir(exist_ok=True)
    output = OUT / "alternating_boundary_layer.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(status="PASS", generated_utc=datetime.now(timezone.utc).isoformat(), rows=len(rows),
        numpy_version=np.__version__, patterns=list(PATTERNS),
        grid=dict(theta=list(THETAS), delta_over_width=list(SCALED_DELTAS), sizes=list(SIZES),
                  exact_zero_included=True, exact_zero_infinite_time_included=True),
        errors=dict(max_scaled_grid_n_positive_asymptotic_abs_error=error,
                    max_scaled_grid_sigma_slope_relative_error=sigma_error,
                    max_orthogonality_error=orth_error, max_projector_error=projector_error,
                    max_exact_zero_positive_occupation_error=zero_error),
        tolerances=dict(scaled_grid_n_positive_abs=5e-6, scaled_grid_sigma_slope_relative=1e-4,
                        orthogonality=1e-12, projector=1e-12, exact_zero_positive_occupation=1e-12),
        derivation={
            "convention": "Primitive-cell momenta, not m=2 supercell momenta. f(q)=1+exp(i*q_x)+exp(-i*q_y), h(f)=[[0,f],[conj(f),0]]. In the paired momentum basis P0=[I;I]/sqrt(2). theta is the same exp(-theta*H) convention as the entropy protocol.",
            "stripe_x": "Pair q=(0,delta), q+Q=(pi,delta), Q=(pi,0): f1=2+exp(-i*delta), f2=exp(-i*delta). Their phase difference is 2*delta/3+O(delta^3). The negative-overlap singular value is |sin(Delta_phi/4)|=|delta|/6+O(delta^3). The two-particle GS amplitude has magnitude |sin(Delta_phi/2)|/2 ~ |delta|/6, and the lowest excited Slater amplitude has magnitude |cos(Delta_phi/2)|/2 ~ 1/2. Its energy exceeds the GS by 2|f2|=2, so the leading excited/GS amplitude ratio is 3*exp(-2*theta)/|delta|. At large theta and small delta this gives n_positive ~ 1/[1+(delta/(3*exp(-2*theta)))^2]. Other Slater sectors provide subleading corrections.",
            "checker": "Pair q=(pi+delta,0), q+Q=(delta,pi) modulo 2pi, Q=(pi,pi): f1=2-exp(i*delta), f2=exp(i*delta). The negative-overlap singular value is |delta|/2+O(delta^3). Both positive energies approach 1 and both negative energies approach -1. The leading rank-one graph singular value is 2*exp(-2*theta)/|delta|, giving n_positive ~ 1/[1+(delta/(2*exp(-2*theta)))^2].",
            "nonzero_graph": "For delta!=0, A=V_minus^dagger P0 is invertible, B=V_plus^dagger P0, X=exp(-theta E_plus) B inv(A) exp(theta E_minus). QR([I;X]) constructs the exact occupied span without unstable large exponential columns. n_positive is the total occupation of the two positive-energy eigenstates.",
            "exact_zero": "No inverse of A is used at delta=0. Stripe x has one supported negative sigma_x branch and one supported positive branch, evolved with explicit normalized weights 1 and exp(-2*theta); its limit occupies energy -3 and +1. Checker has identical sigma_x blocks and an invariant initial occupied projector, with one negative and one positive particle. Thus n_positive=1 for every finite theta and its strict limit. Local overlap rank is 1/2; in the full m=2 block the extra folded sector yields the previously audited rank 3/4.",
            "grid_condition": "Resolving this local momentum layer requires (2*pi/L)/(C*exp(-2*theta)) << 1, C=3 for stripe x and C=2 for checker. Its local crossover time is theta_cross(L)=0.5*log(C*L/(2*pi)); a well-resolved layer needs theta below that value with a sufficient margin. Merely theta/L << 1 does not imply this condition.",
        },
        crossover_theta={name: {str(size): 0.5 * math.log(parameters["width_prefactor"] * size / (2 * math.pi))
                                for size in SIZES} for name, parameters in PATTERNS.items()},
        interpretation=[
            "This is a local, gapped folded-mode mechanism for nonuniform projection near an exact overlap zero. It does not introduce a new Dirac node.",
            "The shrinking momentum width is measured on specified cuts only. Its inverse is not asserted to be the unique correlation length of the full many-body state.",
            "It supplies a reason why theta/L alone need not control finite-size effects for these alternating initial states. No total D, entropy scaling coefficient, or quantitative attribution of the observed D anomaly is derived here.",
            "All calculations are at beta=0 before measurement; finite-beta behavior is not established by this local diagnostic.",
            "The relative-width grid and tolerances test the asymptotic local formula. Nearest-grid rows at L144/180 are diagnostics with their own recorded asymptotic errors, not part of that asymptotic error gate.",
        ],
        files={path.relative_to(BASE).as_posix(): digest(path) for path in (Path(__file__), output)})
    receipt = OUT / "alternating_boundary_layer.json"
    receipt.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(status=summary["status"], rows=len(rows), errors=summary["errors"],
                         files=summary["files"]), indent=2))


if __name__ == "__main__":
    main()
