"""Independently enumerate and audit all six half-filled pure-0/2 m=2 states.

The calculation uses NumPy only and does not import the experiment backend.
Outputs are new data/pure02_m2_* files. No entanglement coefficients are fitted:
the exact initial states cannot access the half-filled periodic ground state.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np


BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
SIZES = (6, 12, 18, 24, 36, 72, 144, 180, 216)
TIMES = (0., 1., 2., 4., 8., 16., 32., 64., 128.)
TOL = 1e-10
OUTPUTS = []


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(name, rows):
    path = DATA / name
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    OUTPUTS.append(path)


def catalogue():
    patterns = []
    for cells in itertools.combinations(range(4), 2):
        bits = "".join("11" if c in cells else "00" for c in range(4))
        pairs = "".join(str(c+1) for c in cells)
        orientation = ("rows" if cells in ((0, 1), (2, 3)) else
                       "columns" if cells in ((0, 2), (1, 3)) else "checkerboard")
        patterns.append(dict(name=f"pure02_m2_cells_{pairs}", bits=bits,
            m=2, class_id=2, n0=2, n1=0, n2=2, occupied_cells=[c+1 for c in cells],
            arrangement=orientation))
    return patterns


def bloch(kx, ky):
    h = np.zeros((8, 8), dtype=complex)
    for y in range(2):
        for x in range(2):
            a = 2*(x+2*y)
            for dx, dy in ((0, 0), (1, 0), (0, -1)):
                xx, yy = x+dx, y+dy
                b = 1+2*(xx % 2+2*(yy % 2))
                value = np.exp(1j*(kx*(xx//2)+ky*(yy//2)))
                h[a, b] += value
                h[b, a] += value.conjugate()
    return h


def occupied(pattern):
    return np.flatnonzero(np.array(list(pattern["bits"])) == "1")


def rank(a, tolerance=TOL):
    return int(np.count_nonzero(np.linalg.svd(a, compute_uv=False) > tolerance))


def grouped(energies):
    first = 0
    while first < len(energies):
        last = first+1
        while last < len(energies) and abs(energies[last]-energies[first]) < 1e-10:
            last += 1
        yield first, last, float(np.mean(energies[first:last]))
        first = last


def accessible_energy(energies, coefficients):
    """Greedy spectral-support rank increments, with no finite-time fitting."""
    previous, result, needed = 0, 0., coefficients.shape[1]
    for first, last, level in grouped(energies):
        current = rank(coefficients[:last])
        assert previous <= current <= needed
        result += (current-previous)*level
        previous = current
        if current == needed:
            break
    assert previous == needed
    return result


def gamma_audit(patterns):
    h = bloch(0., 0.)
    energies, vectors = np.linalg.eigh(h)
    signs = np.array([1., -1., -1., 1.])
    u = np.eye(4)-np.outer(signs, signs)/2
    c_analytic = np.zeros((8, 8))
    c_analytic[::2, ::2] = .5*np.eye(4)
    c_analytic[1::2, 1::2] = .5*np.eye(4)
    c_analytic[::2, 1::2] = -.5*u
    c_analytic[1::2, ::2] = -.5*u
    c_numeric = vectors[:, energies < -TOL] @ vectors[:, energies < -TOL].conj().T
    assert np.max(np.abs(c_analytic-c_numeric)) < 1e-12
    rows = []
    for pattern in patterns:
        f = occupied(pattern)
        c = vectors[f].conj().T
        singular = np.linalg.svd(c[energies < -TOL], compute_uv=False)
        ranks = [rank(c[energies < -TOL], t) for t in (1e-8, 1e-10, 1e-12)]
        assert ranks == [3, 3, 3]
        ea = accessible_energy(energies, c)
        cells = np.array(pattern["occupied_cells"])-1
        empty_cells = np.setdiff1d(np.arange(4), cells)
        cross_rank = rank(u[np.ix_(empty_cells, cells)])
        assert cross_rank == 1 and abs(ea+4) < 1e-12
        rows.append(dict(pattern=pattern["name"],occupied_bits=pattern["bits"],m=2,
            gamma_rank=3,negative_dimension=4,nonpositive_rank=3,required_particles=4,
            gamma_zero_modes=0,ground_energy=-6.,accessible_energy=ea,accessible_excess=ea+6,
            overlap_singular_values=";".join(format(float(s),".17g") for s in singular),
            polar_empty_occupied_rank=cross_rank,rank_stable_1e8_to_1e12=True,reachable=False))
    write_csv("pure02_m2_gamma.csv", rows)
    return dict(energies=energies.tolist(),analytic_projector_max_error=float(np.max(np.abs(c_analytic-c_numeric))),
        proof="In A/B-grouped ordering, T_Gamma=I+Sx+Sy has eigenvalues (3,1,1,-1). Its polar matrix U=sign(T)=I-ss^T/2, s=(1,-1,-1,1). A negative-band frame is [I;-U]/sqrt(2). For any two occupied primitive cells F and their two-cell complement E, U[E,F]=-s_E s_F^T/2 has rank 1. Hence rank(V_negative^dagger P)=2+rank(U[E,F])=3<4. Gamma contains no zero modes; occupying a positive energy +1 instead of a negative energy -1 costs exactly 2 in that block. Every compatible periodic lattice includes Gamma.",
        numerical_rank_tolerances=[1e-8,1e-10,1e-12])


def reachability(patterns):
    rows, bad_rows, probes = [], [], []
    for L in SIZES:
        q = L//2
        totals = {p["name"]: dict(ground=0.,excess=0.,blocked=0,missing_negative=0,
                                  zero_blocks=0,blocked_zero_blocks=0) for p in patterns}
        for iy in range(q):
            for ix in range(q):
                kx, ky = 2*math.pi*ix/q, 2*math.pi*iy/q
                energies, vectors = np.linalg.eigh(bloch(kx,ky))
                negative, nonpositive = energies < -TOL, energies <= TOL
                nneg, nzero = int(np.sum(negative)), int(np.sum(np.abs(energies) <= TOL))
                eg = float(np.sum(energies[:4]))
                for p in patterns:
                    c = vectors[occupied(p)].conj().T
                    rn, rl = rank(c[negative]), rank(c[nonpositive])
                    blocked = rn < nneg or rl < 4
                    total = totals[p["name"]]
                    total["ground"] += eg
                    total["zero_blocks"] += int(nzero > 0)
                    if blocked:
                        ea = accessible_energy(energies,c)
                        excess = ea-eg
                        assert excess > 1e-9
                        total["excess"] += excess
                        total["blocked"] += 1
                        total["missing_negative"] += nneg-rn
                        total["blocked_zero_blocks"] += int(nzero > 0)
                        bad_rows.append(dict(pattern=p["name"],occupied_bits=p["bits"],m=2,L=L,q=q,
                            kx_index=ix,ky_index=iy,kx=kx,ky=ky,negative_dimension=nneg,
                            negative_rank=rn,nonpositive_rank=rl,zero_modes=nzero,
                            required_particles=4,ground_energy=eg,accessible_energy=ea,
                            accessible_excess=excess,is_gamma=(ix==0 and iy==0)))
        for p in patterns:
            total = totals[p["name"]]
            rows.append(dict(pattern=p["name"],occupied_bits=p["bits"],m=2,L=L,
                gamma_rank=3,negative_dimension=4,nonpositive_rank=3,required_particles=4,
                blocked_momenta=total["blocked"],total_momenta=q*q,
                missing_negative_directions=total["missing_negative"],
                momenta_with_zero_modes=total["zero_blocks"],blocked_zero_mode_momenta=total["blocked_zero_blocks"],
                ground_energy=total["ground"],accessible_energy=total["ground"]+total["excess"],
                accessible_excess=total["excess"],excess_per_site=total["excess"]/(2*L*L),reachable=False))
        print("Checked L=",L," blocked=",[totals[p["name"]]["blocked"] for p in patterns],flush=True)
    # Independent off-symmetry probes show that the obstruction is not at all k.
    for kx,ky in ((.41,.77),(1.23,2.2)):
        energies,vectors=np.linalg.eigh(bloch(kx,ky))
        for p in patterns:
            c=vectors[occupied(p)].conj().T
            singular=np.linalg.svd(c[energies < -TOL],compute_uv=False)
            assert rank(c[energies < -TOL]) == 4
            probes.append(dict(pattern=p["name"],kx=kx,ky=ky,negative_dimension=4,
                negative_rank=4,minimum_overlap_singular_value=float(singular[-1])))
    write_csv("pure02_m2_reachability.csv",rows)
    write_csv("pure02_m2_blocked_momenta.csv",bad_rows)
    write_csv("pure02_m2_generic_probes.csv",probes)
    return rows


def real_hamiltonian(L):
    h=np.zeros((2*L*L,2*L*L))
    site=lambda x,y,s: 2*((x%L)+L*(y%L))+s
    for y in range(L):
        for x in range(L):
            a=site(x,y,0)
            for bx,by in ((x,y),(x+1,y),(x,y-1)):
                b=site(bx,by,1)
                h[a,b]+=1
                h[b,a]+=1
    return h


def strict_support(energies,vectors,initial):
    """Energy-adapted real-space reference with exact rank support retained.

    Eigenvectors within degenerate levels are not physically unique. Build the
    right orbital directions from cumulative spectral support and discard only
    entries below a direction's first reachable energy. Reconstruction error
    records the small numerical correction needed to represent exact zeros.
    """
    coefficients=vectors.T@initial
    n,p=coefficients.shape
    right=np.empty((p,0)); starts=[];pivots=[];clustered=energies.copy()
    for first,last,level in grouped(energies):
        clustered[first:last]=level
        if right.shape[1] == p:
            continue
        projected=coefficients[first:last].copy()
        for _ in range(2):
            projected-=(projected@right)@right.T
        _,s,vh=np.linalg.svd(projected,full_matrices=False)
        count=int(np.count_nonzero(s>TOL))
        if count:
            addition=vh[:count].T
            for _ in range(2):
                addition-=right@(right.T@addition)
            addition=np.linalg.qr(addition)[0]
            right=np.column_stack((right,addition))
            starts.extend([first]*count);pivots.extend([level]*count)
    assert right.shape[1] == p
    transformed=coefficients@right;clean=transformed.copy()
    for j,first in enumerate(starts):clean[:first,j]=0
    reconstruction=float(np.linalg.norm(clean-transformed))
    assert reconstruction < 1e-11
    gap=np.maximum(0.,clustered[:,None]-np.array(pivots)[None,:])
    def at(theta):
        if theta==0:return initial.copy()
        weights=(gap==0).astype(float) if math.isinf(theta) else np.exp(-theta*gap)
        return np.linalg.qr(vectors@(clean*weights))[0]
    return at,float(np.sum(pivots)),reconstruction


def projection(patterns,reach):
    L=6;h=real_hamiltonian(L);energies,vectors=np.linalg.eigh(h)
    eg=float(np.sum(energies[:L*L]));ek=(vectors*np.exp(-.25*energies))@vectors.T
    neg=vectors[:,energies < -TOL];pos=vectors[:,energies > TOL];zero=vectors[:,np.abs(energies)<=TOL]
    assert neg.shape[1] == 34 and zero.shape[1] == 4
    rows=[];checks=[]
    for p in patterns:
        F=[2*(x+L*y)+s for y in range(L) for x in range(L) for s in range(2)
           if p["bits"][2*((x%2)+2*(y%2))+s]=="1"]
        initial=np.eye(2*L*L)[:,F]
        limit,ea,reconstruction=strict_support(energies,vectors,initial)
        qinf=limit(math.inf);cinf=qinf@qinf.T
        max_negative=rank(neg.T@initial)
        bloch_result=next(r for r in reach if r["pattern"]==p["name"] and r["L"]==L)
        assert abs(eg-bloch_result["ground_energy"])<1e-9
        assert abs(ea-bloch_result["accessible_energy"])<1e-9
        reference_convergence=float(np.linalg.norm(limit(128)@limit(128).T-cinf)/L)
        assert reference_convergence<1e-10
        direct=initial.copy();previous=0.
        max_small_time_error=0.
        for theta in TIMES:
            for _ in range(round((theta-previous)/.25)):
                direct=np.linalg.qr(ek@direct)[0]
            previous=theta
            reference=limit(theta)
            if theta<=4:
                max_small_time_error=max(max_small_time_error,float(np.linalg.norm(reference@reference.T-direct@direct.T)/L))
            for method,q in (("strict_spectral_support",reference),("direct_qr",direct)):
                c=q@q.T;energy=float(np.trace(q.T@h@q))
                negn=float(np.linalg.norm(neg.T@q)**2)
                rows.append(dict(pattern=p["name"],occupied_bits=p["bits"],L=L,theta=theta,method=method,
                    energy=energy,ground_energy=eg,excess_energy=energy-eg,accessible_energy=ea,
                    accessible_excess=ea-eg,distance_to_accessible_limit=float(np.linalg.norm(c-cinf)/L),
                    reference_convergence=reference_convergence,negative_occupation=negn,
                    maximum_supported_negative_occupation=max_negative,
                    negative_support_violation=max(0.,negn-max_negative),
                    positive_occupation=float(np.linalg.norm(pos.T@q)**2),
                    zero_occupation=float(np.linalg.norm(zero.T@q)**2),
                    orbital_orthogonality_error=float(np.linalg.norm(q.T@q-np.eye(L*L))),
                    eigenstate_residual=float(np.linalg.norm(h@q-q@(q.T@h@q)))))
        energy=float(np.trace(qinf.T@h@qinf));negn=float(np.linalg.norm(neg.T@qinf)**2)
        rows.append(dict(pattern=p["name"],occupied_bits=p["bits"],L=L,theta=math.inf,
            method="strict_spectral_support",energy=energy,ground_energy=eg,excess_energy=energy-eg,
            accessible_energy=ea,accessible_excess=ea-eg,distance_to_accessible_limit=0.,
            reference_convergence=reference_convergence,negative_occupation=negn,
            maximum_supported_negative_occupation=max_negative,negative_support_violation=max(0.,negn-max_negative),
            positive_occupation=float(np.linalg.norm(pos.T@qinf)**2),
            zero_occupation=float(np.linalg.norm(zero.T@qinf)**2),
            orbital_orthogonality_error=float(np.linalg.norm(qinf.T@qinf-np.eye(L*L))),
            eigenstate_residual=float(np.linalg.norm(h@qinf-qinf@(qinf.T@h@qinf)))))
        assert abs(energy-ea)<1e-9 and max_small_time_error<1e-9
        tail=next(r for r in rows if r["pattern"]==p["name"] and r["theta"]==128 and r["method"]=="direct_qr")
        checks.append(dict(pattern=p["name"],realspace_L=L,ground_energy=eg,
            accessible_energy=ea,accessible_excess=ea-eg,
            realspace_vs_bloch_accessible_energy_error=abs(ea-bloch_result["accessible_energy"]),
            spectral_support_reconstruction_error=reconstruction,
            small_time_direct_vs_strict_projector_error=max_small_time_error,
            reference_theta128_vs_infinity_error=reference_convergence,
            direct_theta128_excess=tail["excess_energy"],
            direct_theta128_negative_support_violation=tail["negative_support_violation"],
            direct_theta128_distance_to_accessible_limit=tail["distance_to_accessible_limit"]))
        print(p["name"],"strict excess",ea-eg,"direct128 excess",tail["excess_energy"],
              "forbidden negative occupation",tail["negative_support_violation"],flush=True)
    write_csv("pure02_m2_projection.csv",rows)
    return checks


def main():
    DATA.mkdir(exist_ok=True)
    patterns=catalogue()
    gamma=gamma_audit(patterns)
    reach=reachability(patterns)
    checks=projection(patterns,reach)
    summary=dict(status="PASS",generated_utc=datetime.now(timezone.utc).isoformat(),
        patterns=patterns,sizes=list(SIZES),rank_tolerance=TOL,numpy_version=np.__version__,
        all_six_ground_reachable=False,reachable_patterns=[],
        gamma=gamma,realspace_checks=checks,
        projection_methods={
            "strict_spectral_support":"Independent full real-space spectral-support reference. Numerical null components smaller than rank tolerance are removed before evolution; reconstruction error is recorded. This preserves the mathematically forbidden ground sector and is checked against independently enumerated Bloch-block accessible energies.",
            "direct_qr":"Actual floating-point real-space evolution q <- qr(exp(-0.25 H) q), repeated to each recorded theta. No rank cleaning or symmetry projection. It can amplify roundoff into an exactly forbidden energy sector; a small final ground excess is not evidence of mathematical reachability."},
        interpretation=[
            "All six exact initial configurations are excluded from the finite periodic half-filled ground state by Gamma rank 3<4, on every compatible periodic size. Gamma is gapped and has no zero modes; other zero-energy sectors cannot repair its fixed particle sector.",
            "Gamma alone proves obstruction. Additional symmetry-line momenta are obstructed on larger meshes; counts and coordinates are in the CSV. Generic non-symmetry probes are full rank, so obstruction is not asserted at every momentum.",
            "The strict-support infinite-time state is a configuration-dependent accessible excited state, not the ground state. No equilibrium or nonequilibrium corner coefficients are reported under the previous ground-state protocol.",
            "Total excitation energy and excitation energy per site are distinct. A decreasing energy density does not make the state the exact finite-size ground state or restore the missing many-body overlap.",
            "Direct Float64 QR can leave the exact spectral-support sector because forbidden amplitudes are seeded by roundoff and exponentially amplified. Its departure is measured by negative-band occupation exceeding the maximum allowed by the initial support, not merely by a fit or a small energy difference.",
        ],
        files={path.relative_to(BASE).as_posix():sha(path) for path in OUTPUTS+[Path(__file__)]})
    path=DATA/"pure02_m2_summary.json"
    path.write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    print("PASS: enumerated six patterns; none reaches the true ground state",flush=True)


if __name__=="__main__":
    main()
