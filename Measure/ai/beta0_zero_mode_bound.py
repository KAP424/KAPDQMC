"""Check the finite-rank beta=0 Slater ground-state entropy bound.

The mathematical argument is recorded in the JSON; the small dense checks
illustrate it but are not its proof. No fitted entropy data are read.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[variable] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_beta0_study"


def dense_h(L):
    h = np.zeros((2*L*L,2*L*L))
    for y in range(L):
        for x in range(L):
            a = 2*(x+L*y)
            for dx,dy in ((0,0),(1,0),(0,-1)):
                b = 2*((x+dx)%L+L*((y+dy)%L))+1
                h[a,b] += 1
                h[b,a] += 1
    return h


def rectangle(L,w,h):
    return [2*(x+L*y)+a for y in range(h) for x in range(w) for a in (0,1)]


def entropy(C,indices):
    values = np.linalg.eigvalsh(C[np.ix_(indices,indices)])
    assert values.min()>-2e-12 and values.max()<1+2e-12
    return float(-np.log(values*values+(1-values)*(1-values)).sum())


def run():
    rng = np.random.default_rng(20261022)
    counts,checks = [],[]
    for L in (3,4,5,6,8,9,12):
        h = dense_h(L)
        energy,V = np.linalg.eigh(h)
        negative,zero = energy < -1e-10, abs(energy)<1e-10
        r = int(zero.sum())
        assert r == (4 if L%3==0 else 0)
        k = L*L-int(negative.sum())
        assert k == r//2
        # Separate count from the independently built dense spectrum.
        f = np.array([1+np.exp(2j*np.pi*x/L)+np.exp(-2j*np.pi*y/L) for y in range(L) for x in range(L)])
        analytic_r = 2*int(np.count_nonzero(abs(f)<1e-10))
        assert analytic_r == r
        counts.append(dict(L=L,negative_dimension=int(negative.sum()),zero_dimension=r,zero_occupations=k,
                           bloch_zero_dimension=analytic_r,lowest_nonzero_abs_energy=float(min(abs(energy[~zero])))))
        if not r:
            continue
        W=V[:,zero]
        occupied=V[:,negative]
        Cminus=occupied@occupied.T
        Zref=W[:,:k]@W[:,:k].T
        Cref=Cminus+Zref
        regions={"A":rectangle(L,L//3,L),"C":rectangle(L,L,L//3),"B":rectangle(L,L//3,2*L//3)}
        refS={name:entropy(Cref,R) for name,R in regions.items()}
        refD=(2*refS["A"]+refS["C"])/3-refS["B"]
        for sample in range(5):
            if sample==0:
                Q=np.eye(r,dtype=complex)[:,k:]
            else:
                Q=np.linalg.qr(rng.standard_normal((r,k))+1j*rng.standard_normal((r,k)))[0]
            Z=W@Q@Q.conj().T@W.T
            C=Cminus+Z
            difference=C-Cref
            rank=int(np.count_nonzero(np.linalg.svd(difference,compute_uv=False)>1e-10))
            assert rank<=r
            projected_energy=float(np.trace(h@C).real)
            ground_energy=float(energy[negative].sum())
            assert abs(projected_energy-ground_energy)<1e-10
            row=dict(L=L,sample=sample,zero_dimension=r,difference_rank=rank,
                     projector_residual=float(np.linalg.norm(C@C-C,ord="fro")),
                     ground_energy_error=abs(projected_energy-ground_energy),S2_bound=2*r*math.log(2),D_bound=4*r*math.log(2))
            stateS={}
            for name,R in regions.items():
                block=difference[np.ix_(R,R)]
                blockrank=int(np.count_nonzero(np.linalg.svd(block,compute_uv=False)>1e-10))
                assert blockrank<=r
                stateS[name]=entropy(C,R)
                delta=stateS[name]-refS[name]
                assert abs(delta)<=2*r*math.log(2)+1e-10
                row[f"{name}_difference_rank"]=blockrank
                row[f"{name}_S2_difference"]=delta
            D=(2*stateS["A"]+stateS["C"])/3-stateS["B"]
            row["D_difference"]=D-refD
            assert abs(D-refD)<=4*r*math.log(2)+1e-10
            checks.append(row)
    proof = [
        "Assumptions: same finite one-body Hermitian h_L, fixed half filling, and number-conserving pure Gaussian (Slater determinant) ground states. All logarithms are natural. The measurement strength is beta=0 and theta is the exact ground-state limit, not finite projection time.",
        "Let V_- span strictly negative energies and V_0 span ker h_L of dimension r_L. Every such ground-state projector has C_P=V_- V_-^dagger+V_0 Z_P V_0^dagger, where Z_P is a projector of fixed rank k=N-N_- inside the zero space. Thus rank(C_P-C_ref)<=r_L (and <=2 min(k,r_L-k)).",
        "For any site region R, restriction is C_R=J_R C J_R^dagger. Hence rank(C_{P,R}-C_{ref,R})<=r_L. Both restricted correlation matrices have spectrum in [0,1].",
        "Hermitian rank interlacing implies |N_P(t)-N_ref(t)|<=r_L for every t, with N(t) the eigenvalue counting function. For absolutely continuous f with f(0)=f(1)=0, integration by parts yields Tr f(C_{P,R})-Tr f(C_{ref,R})=-integral_0^1 f'(t)[N_P(t)-N_ref(t)]dt.",
        "For second Renyi entropy f(t)=-ln[t^2+(1-t)^2], f grows from 0 to ln2 at t=1/2 and decreases back to 0. Its total variation is 2 ln2. Consequently |S2(R;P)-S2(R;ref)|<=2 r_L ln2.",
        "For the same three fixed regions and D=(2 S_A+S_C)/3-S_B, the sum of absolute weights is 2. The triangle inequality gives |D_P(L)-D_ref(L)|<=4 r_L ln2. This bound itself does not assume an area-law cancellation.",
        "For this untwisted periodic honeycomb hopping, the 2x2 Bloch off-diagonal element is f(k)=1+exp(i k_x)+exp(-i k_y). Its only zeros are k=(2pi/3,2pi/3) and (4pi/3,4pi/3), modulo 2pi. Each zero contributes a two-dimensional single-particle zero space. Thus r_L=4 when 3 divides L and r_L=0 otherwise, for L>=3 and nonzero hopping. Supercell folding does not increase the physical zero-mode count.",
        "On any common sequence of sizes with uniformly bounded r_L, [D_P(L)-D_ref(L)]/ln L tends to zero. If D_ref(L)/ln L has a finite limit c (for example after the common area term cancels and a log asymptotic expansion exists), then D_P(L)/ln L has the same limit. If either D also contains a surviving common linear area term, the rank bound does not by itself make D/ln L finite.",
        "The bounded difference may vary with L. It is not necessarily one constant and need not be numerically small. Different finite-size fitted c values therefore cannot alone establish distinct beta=0 asymptotic ground-state corner coefficients. Conversely this argument proves neither that a log term exists nor that a specific finite window is asymptotic.",
        "This proof does not cover beta>0 filtered states, finite-theta states, inaccessible excited limits, Hamiltonians with extensive zero-mode degeneracy, or non-Gaussian superpositions solely through their two-point matrix. It makes no claim that every initial pattern admits extraction of the same coefficient in a finite 1<<theta<<L window."
    ]
    result=dict(status="PASS",generated_utc=datetime.now(timezone.utc).isoformat(),scope="Beta=0 exact Slater ground-state finite-rank theorem and small dense illustrations",
                proof_steps=proof,conservative_uniform_bounds=dict(r_max=4,regional_S2=8*math.log(2),D=16*math.log(2)),
                zero_mode_counts=counts,dense_checks=checks,
                counts=dict(sizes_checked=len(counts),independent_zero_mode_states=len(checks)),
                maxima=dict(projector_residual=max(r["projector_residual"] for r in checks),
                            ground_energy_error=max(r["ground_energy_error"] for r in checks),
                            region_entropy_difference=max(abs(r[f"{region}_S2_difference"]) for r in checks for region in ("A","B","C")),
                            D_difference=max(abs(r["D_difference"]) for r in checks)),
                files={Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                evidence_limit="Finite small-system checks illustrate the self-contained rank/counting-function proof; they do not replace it or read entropy-fit outputs.")
    (OUT/"zero_mode_bound.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:result[k] for k in ("status","counts","conservative_uniform_bounds","maxima","zero_mode_counts")},indent=2))


if __name__=="__main__":
    run()
