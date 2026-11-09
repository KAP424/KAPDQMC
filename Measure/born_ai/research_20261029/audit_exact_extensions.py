"""Read-only audit of the exhaustive data, historical L3 means, and FFT kernels.

Also stress-tests the stronger outcome-wise bounded-filter statement on random
complex Slater states. These tests accompany, rather than replace, min-max proof.
"""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import math
import os
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/"exact_strata"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def generalized_eigvals(A,B):
    chol=np.linalg.cholesky(B)
    left=np.linalg.solve(chol,A)
    white=np.linalg.solve(chol,left.conj().T).conj().T
    return np.linalg.eigvalsh((white+white.conj().T)/2)

spec=importlib.util.spec_from_file_location("baseline",HERE/"compute_corner_baseline.py")
baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)
rows=[];fftchecks=[]
for L in (2,3):
    summary=json.loads((OUT/f"L{L}_summary.json").read_text())
    prior=np.load(OUT/f"L{L}_conditional_probabilities.npz")
    G0=prior["G0"]; V0=np.array(summary["initial_V"])
    K=baseline.kernels(L)
    cells=np.arange(2*L*L)//2
    dx=(cells[:,None]%L-cells[None,:]%L)%L
    dy=(cells[:,None]//L-cells[None,:]//L)%L
    reconstructed=np.empty(G0.shape,complex)
    for a in (0,1):
        for b in (0,1):
            reconstructed[a::2,b::2]=K[a,b,dy[a::2,b::2],dx[a::2,b::2]]
    error=float(np.max(abs(reconstructed-G0)))
    den=2 if L==2 else 3
    dv=baseline.corners(K,den)[0]
    assert error<1e-12
    assert abs(dv-summary["initial_DV_full"])<1e-12
    fftchecks.append(dict(L=L,projector_max_error=error,DV_error=abs(dv-summary["initial_DV_full"])))
    for beta in (3,4,5):
        data=np.load(OUT/f"L{L}_beta{beta}.npz")
        ratio=data["observables"][:,4:7]/V0
        lower=math.exp(-4*beta)
        assert ratio.min()>=lower-1e-13
        assert ratio.max()<=1/lower+1e-12
        rows.append(dict(L=L,beta=beta,min_outcome_V_over_V0=float(ratio.min()),
                         max_outcome_V_over_V0=float(ratio.max()),lower=lower,upper=1/lower))

oldpath=HERE.parent/"data/born_scaling/exact_L3_mu1e-8_beta3p0_trajectories.csv"
oldmeans=np.zeros(4); oldsum=0.
with oldpath.open() as f:
    for row in csv.DictReader(f):
        prob=float(row["probability"]);oldsum+=prob
        oldmeans+=prob*np.array([float(row[k]) for k in ("S_A","S_B","S_C","D")])
new=json.loads((OUT/"L3_summary.json").read_text())["cases"][0]["mean"]
meanerr=float(np.max(abs(oldmeans-np.array([new[k] for k in ("SA","SB","SC","D")]))))
assert meanerr<1e-10

rng=np.random.default_rng(2026102907)
maxviolation=0.; trials=0
for N,n,r in ((8,4,4),(12,4,6),(18,9,9)):
    for beta in (.1,1.,3.):
        q=math.exp(-4*beta)
        for repeat in range(100):
            Q=np.linalg.qr(rng.normal(size=(N,n))+1j*rng.normal(size=(N,n)))[0]
            A0=Q[:r].conj().T@Q[:r];B0=Q[r:].conj().T@Q[r:]
            w=np.exp(rng.uniform(-2*beta,0,size=N))
            if repeat%2==0:
                w=np.where(rng.random(N)<.5,math.exp(-2*beta),1.)
            A=(Q[:r]*w[:r,None]).conj().T@(Q[:r]*w[:r,None])
            B=(Q[r:]*w[r:,None]).conj().T@(Q[r:]*w[r:,None])
            odds0=generalized_eigvals(A0,B0)
            odds=generalized_eigvals(A,B)
            relative=odds/odds0
            violation=max(0.,float(q-relative.min()),float(relative.max()-1/q))
            maxviolation=max(maxviolation,violation)
            # Numerical generalized eigensolvers are sensitive for nearly singular B;
            # allow only a relative 1e-7 roundoff tolerance on the upper endpoint.
            assert relative.min()>=q*(1-1e-7)
            assert relative.max()<=(1/q)*(1+1e-7)
            P=np.linalg.qr(Q*w[:,None])[0]
            lam=np.linalg.eigvalsh(P[:r].conj().T@P[:r])
            lam0=np.linalg.eigvalsh(A0)
            V=float((lam*(1-lam)).sum());V0=float((lam0*(1-lam0)).sum())
            assert q*V0-1e-12<=V<=V0/q+1e-12
            trials+=1

receipt=dict(status="PASS",exhaustive_outcome_filter_checks=rows,
             random_complex_minmax_checks=dict(trials=trials,max_absolute_interval_violation=maxviolation),
             independent_FFT_dense_checks=fftchecks,
             historical_L3_beta3=dict(path=str(oldpath.relative_to(HERE.parent)),sha256=sha(oldpath),
                                      probability_sum_error=abs(oldsum-1),mean_max_error=meanerr),
             known_ground_corner_coefficient=(4+2*math.pi/(3*math.sqrt(3)))/(8*math.pi**2),
             benchmark_receipt_sha256=sha(OUT/"receipt.json"),
             FFT_source_sha256=sha(HERE/"compute_corner_baseline.py"),
             audit_source_sha256=sha(__file__))
(OUT/"extensions_audit.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt,indent=2))
