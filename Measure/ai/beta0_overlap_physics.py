"""Low-cost beta=0 initial-boundary diagnostics, independent of entropy code.

Scan periodic product configurations without reading any entropy or fitted
corner coefficient.  Fourier reconstruction of the chiral Bloch eigenframe
separates gapped overlap zeros from Dirac-neighbourhood effects.  Finite scans
are candidate screens, never proofs that a continuum overlap is bounded.
"""
from __future__ import annotations
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "analysis_beta0_study"
TWOPI = 2 * math.pi
GRID_SIZES = (31, 61)
GAP_CUT = 0.3
RADII = (1e-2, 1e-3, 1e-4, 1e-5)
ANGLE_COUNT = 48
THETAS = (2.0, 4.0, 7.0, 12.0)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def catalogue():
    sources = [BASE / "data/patterns.csv", BASE / "data/arrangement_selection.csv",
               BASE / "occupancy_classes_patterns.jl", BASE / "second_class_extra_patterns.jl",
               BASE / "irregular_m3_patterns.jl"]
    patterns = {r["pattern"]: dict(name=r["pattern"], m=int(r["m"]), bits=r["occupied_bits"])
                for path in sources[:2] for r in table(path)}
    text = sources[2].read_text(encoding="utf-8")
    for name, m, bits in re.findall(r'\("([a-zA-Z0-9_]+)",\s*"[a-z]+",\s*([1-4]),\s*"([01]+)"', text):
        patterns[name] = dict(name=name, m=int(m), bits=bits)
    text = sources[3].read_text(encoding="utf-8")
    for name, m, bits in re.findall(r'\("([a-zA-Z0-9_]+)",\s*([1-4]),\s*"([01]+)"', text):
        if name == "pure02_alternating_x":
            patterns.pop("full_empty_x")
        if name == "pure02_alternating_checker":
            patterns.pop("full_empty_checker")
        patterns[name] = dict(name=name, m=int(m), bits=bits)
    bits = re.search(r'bits = "([01]+)"', sources[4].read_text(encoding="utf-8"))[1]
    patterns["near02_irregular3"] = dict(name="near02_irregular3", m=3, bits=bits)
    for p in patterns.values():
        p["F"] = np.flatnonzero(np.array(list(p["bits"])) == "1")
        cells = [p["bits"][i:i+2].count("1") for i in range(0, len(p["bits"]), 2)]
        p["class_id"] = 1 if all(n == 1 for n in cells) else 2 if all(n != 1 for n in cells) else 3
        assert len(p["F"]) == p["m"]**2
    return patterns, sources


def frame(m, k):
    """Analytic Fourier bands in original interleaved A/B site order."""
    sites = np.array([(x, y) for y in range(m) for x in range(m)])
    momenta = (np.asarray(k)[None, :] + TWOPI * sites) / m
    w = np.exp(1j * sites @ momenta.T) / m
    f = 1 + np.exp(1j * momenta[:, 0]) + np.exp(-1j * momenta[:, 1])
    eps = np.abs(f)
    phase = np.conj(f / eps)
    minus = np.empty((2*m*m, m*m), complex)
    plus = np.empty_like(minus)
    minus[0::2] = plus[0::2] = w / math.sqrt(2)
    minus[1::2] = -w * phase[None, :] / math.sqrt(2)
    plus[1::2] = -minus[1::2]
    return eps, minus, plus


def dense_h(m, k):
    h = np.zeros((2*m*m, 2*m*m), complex)
    for y in range(m):
        for x in range(m):
            a = 2*(x + m*y)
            for dx, dy in ((0, 0), (1, 0), (0, -1)):
                xx, yy = x+dx, y+dy
                b = 2*(xx % m + m*(yy % m)) + 1
                z = np.exp(1j*(k[0]*(xx//m) + k[1]*(yy//m)))
                h[a, b] += z
                h[b, a] += z.conjugate()
    return h


def sample(p, k):
    eps, vm, vp = frame(p["m"], k)
    a = vm[p["F"]].conj().T
    return float(np.linalg.svd(a, compute_uv=False)[-1]), float(2*eps.min()), a, eps, vm, vp


def minimize(fun, seed, step=.12, maxiter=500):
    """Deterministic two-dimensional Nelder-Mead; independent of SciPy."""
    x = np.array([seed, np.asarray(seed)+[step, 0], np.asarray(seed)+[0, step]], float)
    f = np.array([fun(v) for v in x])
    count = 3
    for iteration in range(maxiter):
        order = np.argsort(f); x, f = x[order], f[order]
        if np.max(np.linalg.norm(x[1:]-x[0], axis=1)) < 2e-11:
            break
        center = x[:2].mean(axis=0)
        xr = center + (center-x[2]); fr = fun(xr); count += 1
        if fr < f[0]:
            xe = center + 2*(xr-center); fe = fun(xe); count += 1
            x[2], f[2] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < f[1]:
            x[2], f[2] = xr, fr
        else:
            xc = center + .5*((xr-center) if fr < f[2] else (x[2]-center))
            fc = fun(xc); count += 1
            if fc < min(fr, f[2]):
                x[2], f[2] = xc, fc
            else:
                x[1:] = x[0]+.5*(x[1:]-x[0])
                f[1:] = [fun(v) for v in x[1:]]; count += 2
    index = int(np.argmin(f))
    return x[index], float(f[index]), iteration+1, count


def grid_scans(patterns):
    rows, candidates = [], {}
    for m in sorted({p["m"] for p in patterns.values()}):
        group = [p for p in patterns.values() if p["m"] == m]
        for n in GRID_SIZES:
            records = {p["name"]: [] for p in group}
            for iy in range(n):
                for ix in range(n):
                    k = TWOPI*np.array([(ix+.371)/n, (iy+.193)/n])
                    eps, vm, _ = frame(m, k)
                    gap = float(2*eps.min())
                    if gap < GAP_CUT: continue
                    for p in group:
                        sigma = float(np.linalg.svd(vm[p["F"]].conj().T, compute_uv=False)[-1])
                        records[p["name"]].append((sigma, k[0], k[1], gap))
            for p in group:
                rec = sorted(records[p["name"]]); s = np.array([r[0] for r in rec])
                best = rec[0]
                rows.append(dict(pattern=p["name"], m=m, class_id=p["class_id"], grid_n=n,
                    kept_gapped_points=len(rec), gap_cut=GAP_CUT, sigma_min=best[0],
                    sigma_q01=float(np.quantile(s, .01)), sigma_median=float(np.median(s)),
                    count_sigma_below_1e_3=int(np.count_nonzero(s < 1e-3)),
                    min_kx=best[1], min_ky=best[2], gap_at_min=best[3]))
                if n == max(GRID_SIZES):
                    seeds = [np.array(rec[0][1:3])]
                    for r in rec[1:]:
                        k = np.array(r[1:3])
                        if all(np.linalg.norm((k-old+math.pi) % TWOPI-math.pi) > .3 for old in seeds):
                            seeds.append(k)
                        if len(seeds) == 3: break
                    candidates[p["name"]] = seeds
            print("grid", m, n, "patterns", len(group), flush=True)
    return rows, candidates


def refined_nodes(patterns, candidates):
    rows, dynamics = [], []
    for name, p in patterns.items():
        def objective(k):
            sigma, gap, *_ = sample(p, k)
            return sigma if gap >= GAP_CUT else 1+GAP_CUT-gap
        for seed_index, seed in enumerate(candidates[name]):
            k, _, iterations, calls = minimize(objective, seed)
            sigma, gap, a, eps, vm, vp = sample(p, k)
            u, _, vh = np.linalg.svd(a)
            left, right = u[:, -1], vh[-1].conj()
            derivative = []
            for d in np.eye(2)*1e-5:
                ap = sample(p, k+d)[2]; am = sample(p, k-d)[2]
                derivative.append(np.vdot(left, ((ap-am)/(2e-5))@right))
            jacobian = np.array([np.real(derivative), np.imag(derivative)])
            jacobian_s = np.linalg.svd(jacobian, compute_uv=False)
            row = dict(pattern=name, m=p["m"], class_id=p["class_id"], seed_index=seed_index,
                kx=float(k[0] % TWOPI), ky=float(k[1] % TWOPI), sigma_min=sigma, gap=gap,
                numerical_zero_witness=sigma < 1e-8 and gap > GAP_CUT-1e-10,
                jacobian_sigma_max=float(jacobian_s[0]), jacobian_sigma_min=float(jacobian_s[1]),
                jacobian_ratio=float(jacobian_s[1]/jacobian_s[0]) if jacobian_s[0] else None,
                iterations=iterations, function_calls=calls)
            rows.append(row)
            # Offset from the candidate: exact null points must not be inverted.
            for displacement in (1e-2, 1e-3, 1e-4):
                probe = k + np.array([displacement, .379*displacement])
                s, g, aa, ee, vv, vp2 = sample(p, probe)
                if s < 1e-12: continue
                bb = vp2[p["F"]].conj().T
                x0 = np.linalg.solve(aa.T, bb.T).T
                for theta in THETAS:
                    x = x0*np.exp(-theta*(ee[:, None]+ee[None, :]))
                    ss = np.linalg.svd(x, compute_uv=False)
                    npos = float(np.sum(ss**2/(1+ss**2)))
                    dynamics.append(dict(pattern=name, seed_index=seed_index, displacement=displacement,
                        kx=float(probe[0] % TWOPI), ky=float(probe[1] % TWOPI), gap=g,
                        sigma_min=s, theta=theta, graph_norm=float(ss[0]), positive_occupation=npos))
        print("refined", name, min(r["sigma_min"] for r in rows if r["pattern"] == name), flush=True)
    return rows, dynamics


def dirac_scan(patterns):
    rows = []
    for p in patterns.values():
        for valley in (-1, 1):
            origin = np.full(2, valley*TWOPI/3)
            for radius in RADII:
                for j in range(ANGLE_COUNT):
                    angle = TWOPI*j/ANGLE_COUNT
                    q = origin+radius*np.array([math.cos(angle), math.sin(angle)])
                    k = p["m"]*q
                    sigma, gap, *_ = sample(p, k)
                    rows.append(dict(pattern=p["name"], m=p["m"], class_id=p["class_id"], valley=valley,
                        physical_radius=radius, angle_index=j, angle=angle, kx=float(k[0] % TWOPI),
                        ky=float(k[1] % TWOPI), sigma_min=sigma, gap=gap,
                        logarithmic_overlap_cost=-math.log(max(sigma, 1e-300)),
                        conservative_graph_time_1pct=math.log(max(1., math.sqrt(max(0.,sigma**-2-1))/.01))/gap))
    return rows


def golden_minimum(fun, left, right):
    ratio = (math.sqrt(5)-1)/2
    a, b = right-ratio*(right-left), left+ratio*(right-left)
    fa, fb = fun(a), fun(b)
    for _ in range(90):
        if fa < fb:
            right, b, fb = b, a, fa
            a = right-ratio*(right-left); fa = fun(a)
        else:
            left, a, fa = a, b, fb
            b = left+ratio*(right-left); fb = fun(b)
    return (a, fa) if fa < fb else (b, fb)


def low_energy_boundary(patterns, nodes):
    """Eliminate filled high-negative bands, then identify the Dirac boundary."""
    summaries, angular, collapses = [], [], []
    blocked = {r["pattern"] for r in nodes if r["numerical_zero_witness"]}
    for name, p in patterns.items():
        for valley in (-1, 1):
            k = p["m"]*np.full(2, valley*TWOPI/3)
            h = dense_h(p["m"], k); e, v = np.linalg.eigh(h)
            negative, zero = e < -1e-8, np.abs(e) < 1e-8
            v0, vn = v[:, zero], v[:, negative]
            a = vn[p["F"]].conj().T
            _, s, vh = np.linalg.svd(a, full_matrices=True)
            rank = int(np.count_nonzero(s > 1e-10))
            r = vh[rank:].conj().T
            raw = v0[p["F"]].conj().T @ r
            u0, s0, _ = np.linalg.svd(raw, full_matrices=False)
            rank0 = int(np.count_nonzero(s0 > 1e-10))
            need = int(np.sum(zero))//2
            row = dict(pattern=name, m=p["m"], class_id=p["class_id"], valley=valley,
                zero_dimension=int(np.sum(zero)), high_negative_dimension=int(np.sum(negative)),
                high_negative_overlap_rank=rank, zero_support_rank=rank0,
                boundary_status="regular_high_energy_elimination" if rank == np.sum(negative) and rank0 == need else "rank_obstructed_at_Dirac",
                zero_support_sigma_min=float(s0[-1]) if len(s0) else None,
                sublattice_polarization=None, mass_anticommutator_max=None,
                dirac_angular_sigma_min=None, minimizing_angle=None,
                two_dimensional_formula_min=None, two_dimensional_formula_error=None)
            if row["boundary_status"] != "regular_high_energy_elimination":
                summaries.append(row); continue
            qb = u0[:, :need]; cb = qb@qb.conj().T; mass = 2*cb-np.eye(len(cb))
            alpha = []
            for d in np.eye(2)*1e-5:
                alpha.append(p["m"]*v0.conj().T@(dense_h(p["m"],k+d)-dense_h(p["m"],k-d))@v0/(2e-5))
            def angular_overlap(angle):
                hd = math.cos(angle)*alpha[0]+math.sin(angle)*alpha[1]
                ed, vd = np.linalg.eigh(hd)
                return float(np.linalg.svd(vd[:, ed < -1e-8].conj().T@qb, compute_uv=False)[-1])
            angle_grid = np.linspace(0, TWOPI, 96, endpoint=False)
            values = [angular_overlap(angle) for angle in angle_grid]
            for angle, sigma in zip(angle_grid, values):
                angular.append(dict(pattern=name, valley=valley, angle=float(angle), sigma_min=sigma))
            j = int(np.argmin(values)); step = TWOPI/len(angle_grid)
            angle, value = golden_minimum(angular_overlap, angle_grid[j]-step, angle_grid[j]+step)
            sz = np.diag(np.tile([1, -1], p["m"]**2))
            pol = float(np.trace(v0@cb@v0.conj().T@sz).real)/need
            anti = max(float(np.linalg.norm(mass@aa+aa@mass)/(2*np.linalg.norm(aa))) for aa in alpha)
            formula = math.sqrt(max(0., (1-math.sqrt(max(0., 1-pol*pol)))/2)) if len(cb) == 2 else None
            row.update(sublattice_polarization=pol, mass_anticommutator_max=anti,
                dirac_angular_sigma_min=value, minimizing_angle=float(angle % TWOPI),
                two_dimensional_formula_min=formula,
                two_dimensional_formula_error=abs(formula-value) if formula is not None else None)
            summaries.append(row)
            if name in blocked: continue
            # Test the proposed local scaling kernel without any real-space
            # entropy. The high-band frame is held at the Dirac momentum.
            for direction in (.37, math.pi/4, 2.0):
                vector = np.array([math.cos(direction), math.sin(direction)])
                hd = alpha[0]*vector[0]+alpha[1]*vector[1]
                ed, vd = np.linalg.eigh(hd)
                for scaled_time in (.5, 1., 2.):
                    evolved = vd@(np.exp(-scaled_time*ed)[:, None]*(vd.conj().T@qb))
                    qe = np.linalg.qr(evolved)[0]; ce = qe@qe.conj().T
                    expected = vn@vn.conj().T+v0@ce@v0.conj().T
                    for radius in (1e-2, 1e-3, 1e-4):
                        probe = k+p["m"]*radius*vector
                        sigma, _, aa, eps, vm, vp = sample(p, probe)
                        assert sigma > 1e-10
                        x0 = np.linalg.solve(aa.T, (vp[p["F"]].conj().T).T).T
                        theta = scaled_time/radius
                        x = x0*np.exp(-theta*(eps[:, None]+eps[None, :]))
                        q = np.hstack((vm, vp))@np.linalg.qr(np.vstack((np.eye(p["m"]**2),x)))[0]
                        c = q@q.conj().T
                        collapses.append(dict(pattern=name, valley=valley, angle=direction,
                            physical_radius=radius, theta=theta, radius_times_theta=scaled_time,
                            boundary_zero_block_error=float(np.linalg.norm(v0.conj().T@c@v0-ce)),
                            full_local_projector_error=float(np.linalg.norm(c-expected)),
                            sigma_min=sigma))
    return summaries, angular, collapses


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)


def main():
    patterns, sources = catalogue(); OUT.mkdir(exist_ok=True)
    errors = []
    for m in (1, 2, 3, 4):
        k = np.array([.617, 2.213]); eps, vm, vp = frame(m, k); h = dense_h(m, k)
        errors += [float(np.linalg.norm(h@vm+vm*eps)), float(np.linalg.norm(h@vp-vp*eps)),
                   float(np.linalg.norm(vm.conj().T@vm-np.eye(m*m)))]
    assert max(errors) < 1e-12
    grids, seeds = grid_scans(patterns)
    nodes, dynamics = refined_nodes(patterns, seeds)
    dirac = dirac_scan(patterns)
    boundaries, angles, collapses = low_energy_boundary(patterns, nodes)
    collapse_groups = {}
    for row in collapses:
        key = (row["pattern"], row["valley"], row["angle"], row["radius_times_theta"])
        collapse_groups.setdefault(key, []).append(row)
    convergence_ratios = []
    for group in collapse_groups.values():
        group.sort(key=lambda r:r["physical_radius"], reverse=True)
        convergence_ratios.extend(right["full_local_projector_error"]/left["full_local_projector_error"]
                                  for left, right in zip(group, group[1:]))
    formula_errors = [r["two_dimensional_formula_error"] for r in boundaries
                      if r["two_dimensional_formula_error"] is not None]
    assert len(collapses) == 918 and len(collapse_groups) == 306
    assert max(convergence_ratios) < .11 and min(convergence_ratios) > .09
    assert max(formula_errors) < 1e-7
    outputs = []
    for label, rows in (("grid_summary", grids), ("gapped_minima", nodes),
                        ("gapped_mode_dynamics", dynamics), ("dirac_scan", dirac),
                        ("dirac_boundary", boundaries), ("dirac_boundary_angles", angles),
                        ("dirac_scaling_kernel", collapses)):
        path = OUT / ("physics_"+label+".csv"); write_csv(path, rows); outputs.append(path)
    overview = []
    for name, p in patterns.items():
        g = next(r for r in grids if r["pattern"] == name and r["grid_n"] == 61)
        n = min((r for r in nodes if r["pattern"] == name), key=lambda r:r["sigma_min"])
        d = [r for r in dirac if r["pattern"] == name]
        boundary = next(r for r in boundaries if r["pattern"] == name and r["valley"] == 1)
        mins = [min(r["sigma_min"] for r in d if r["physical_radius"] == radius) for radius in RADII]
        overview.append(dict(pattern=name, m=p["m"], class_id=p["class_id"],
            gapped_grid_sigma_min=g["sigma_min"], gapped_refined_sigma_min=n["sigma_min"],
            gapped_refined_kx=n["kx"], gapped_refined_ky=n["ky"], gapped_refined_gap=n["gap"],
            gapped_zero_witness=n["numerical_zero_witness"],
            dirac_min_sigma_radius_1e_2=mins[0], dirac_min_sigma_radius_1e_5=mins[-1],
            dirac_radial_power_estimate=math.log(max(mins[-1],1e-300)/max(mins[0],1e-300))/math.log(1e-3),
            effective_dirac_boundary_sigma_min=boundary["dirac_angular_sigma_min"],
            effective_dirac_sublattice_polarization=boundary["sublattice_polarization"],
            effective_dirac_mass_anticommutator_max=boundary["mass_anticommutator_max"]))
    overview_file = OUT / "physics_pattern_summary.csv"; write_csv(overview_file, overview); outputs.append(overview_file)
    result = dict(status="PASS", generated_utc=datetime.now(timezone.utc).isoformat(),
        patterns=[{k:v for k,v in p.items() if k != "F"} for p in patterns.values()],
        controls=dict(grid_sizes=list(GRID_SIZES), gapped_minimum_gap=GAP_CUT,
            grid_offsets=[.371,.193], local_minimization_seeds=3, dirac_physical_radii=list(RADII),
            dirac_angle_count=ANGLE_COUNT, times=list(THETAS), entropy_data_read=False),
        max_analytic_eigenframe_residual=max(errors),
        low_energy_kernel_validation=dict(rows=len(collapses), independent_parameter_groups=len(collapse_groups),
            protocol="rho=1e-2,1e-3,1e-4 with rho*theta fixed at .5,1,2, three directions and two valleys; all 17 candidates without detected gapped zero witnesses, including uniform B control",
            error_ratio_for_tenfold_radius_reduction_min=min(convergence_ratios),
            error_ratio_for_tenfold_radius_reduction_max=max(convergence_ratios),
            conclusion="The full local projector approaches the derived Dirac-boundary kernel linearly in rho on every tested group. This validates a local scaling limit, not finite-theta entropy scaling.",
            max_two_dimensional_boundary_formula_error=max(formula_errors),
            two_dimensional_formula_note="The roughly 1e-8 error for uniform A/B arises from square-root sensitivity to roundoff in |n_z|=1; all nonuniform two-dimensional checks agree to roundoff."),
        derivations={
            "gapped_graph": "A=V_minus^dagger P, B=V_plus^dagger P. Where A is invertible, X(theta)=exp(-theta E_plus) B inv(A) exp(theta E_minus). The norm bound exp(-Delta*theta)*sqrt(s_min^-2-1) exposes an initial-state-dependent prefactor. A uniform positive lower bound on s away from Dirac nodes gives a uniform microscopic projection time on that compact momentum region.",
            "uniform": "For uniform A/B, every negative-band overlap is exactly 1/sqrt(2). In an unfolded 2x2 block, C_theta=[I-tanh(2*epsilon*theta)H/epsilon +/-sech(2*epsilon*theta)sigma_z]/2. The only critical crossover variable is epsilon*theta; epsilon~v|q-K| gives the ordinary theta length scale. No extra gapped zero-overlap layer exists.",
            "minor": "For chiral H=[0,T;T^dagger,0], polar unitary U=T(T^dagger T)^-1/2 gives V_minus=[I;-U^dagger]/sqrt(2). At half filling, the overlap determinant reduces, up to a nonzero factor and sign, to the minor of U linking occupied B sites to empty A sites. Thus the spatial arrangement, not only A/B counts, controls extra zeros.",
            "dirac": "If gap~v*r and the relevant overlap behaves as r^p, the graph bound contains r^-p exp(-v*r*theta). This can add logarithmic overlap costs to projection times compared with a bounded Dirac boundary overlap. It does not by itself determine D or prove a log-log entropy correction.",
            "effective_boundary": "At an exact folded Dirac momentum, remove the filled high-negative directions: R spans ker(V_high_negative^dagger P). The low-energy initial boundary is Q_b=orth(V_zero^dagger P R), provided the high-negative rank and zero-support rank are correct. Evolving Q_b with the projected linear Dirac Hamiltonian describes the rho->0, theta->infinity, rho*theta fixed local kernel. The script tests this limit against the full analytic Bloch evolution; it does not compute entropy.",
            "boundary_mass": "M=2 Q_b Q_b^dagger-I is a flattened initial-boundary matrix, not an added physical Hamiltonian term. For uniform A/B it anticommutes with both Dirac kinetic matrices. More general regular boundaries need not anticommute. In a two-dimensional zero space with sublattice polarization n_z, the smallest negative-Dirac overlap over all directions is sqrt((1-sqrt(1-n_z^2))/2); n_z=0 is the angular-obstruction case. Four-dimensional valley-mixed zero spaces require the full angular matrix test.",
            "valley_folding_symmetry": "The primitive Dirac valleys K_plus/minus=(+/-2pi/3,+/-2pi/3) fold to the same m-by-m supercell momentum exactly when m is divisible by 3. Otherwise primitive translation by the supercell separates the two two-dimensional valley blocks. For m divisible by 3, a periodic initial configuration can mix the valleys in a four-dimensional zero space; its mean sublattice polarization is not a sufficient boundary-regularity criterion.",
            "conditional_scaling_argument": "If overlaps away from Dirac points have a uniform positive lower bound and the effective Dirac boundary is nondegenerate in every direction, then there is no extra divergent gapped projection scale, and the low-energy kernel depends on rho*theta after a finite initial-state crossover. Modes at separations a<<r<<v*theta can then approach the GS critical kernel; obtaining the same corner-log coefficient is the natural scaling argument. This is conditional physical reasoning, not a rigorous theorem for the full Renyi entropy or proof from a finite grid.",
        },
        interpretation=[
            "The two shifted momentum grids and local minimizers do not prove absence of continuum zeros. Values below 1e-8 at a gapped point are numerical zero witnesses, not exact symbolic theorems.",
            "A finite-size ground-reachability audit may miss a continuum zero at an incommensurate momentum. A near-zero also produces large but finite crossover time without any true zero.",
            "The derivative Jacobian tests local first-order complex overlap structure only. It does not alone certify a global zero line or a topological defect.",
            "Dirac angular/radial scans are candidate diagnostics. Avoid labeling an overlap minimum caused by the physical vanishing energy gap as an extra gapped obstruction.",
            "A concrete finite-sampling caution: the 48-direction full-overlap scan misses angular zeros of some blocked m2 stripes, while the effective-boundary angular minimizer finds them. A nonzero sampled minimum must never be promoted to a continuum theorem.",
            "All conclusions concern beta=0 before measurement. No entropy is computed, and no pattern is declared to have a certified GS corner coefficient from this scan alone.",
            "Uniform A/B is the safest analytic reference. Other patterns can only be prioritized for actual size-plateau and time-window tests after these spectral diagnostics.",
        ],
        primary_reference=dict(url="https://www.nature.com/articles/s41467-026-75645-x",
            scope="The paper proposes a finite-size scaling ansatz and motivates its short-time limit by xi~tau^(1/z). Its free-fermion experiment starts from an AFM product state and checks size-independent plateaus. It is not a proof of this single-scale condition for arbitrary periodic product configurations."),
        files={path.relative_to(BASE).as_posix():sha(path) for path in [Path(__file__)]+sources+outputs})
    (OUT / "physics_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print("PASS",len(patterns),"patterns",len(dirac),"Dirac probes",flush=True)
    for r in overview: print(json.dumps(r), flush=True)


if __name__ == "__main__": main()
