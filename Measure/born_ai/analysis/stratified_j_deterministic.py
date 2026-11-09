"""Exact small-system stratification variance, with no Monte Carlo sampling."""
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import tomllib
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE.parent


def main():
    paths=[BASE/"data/local_orbitals_L2.bin",
           BASE/"data/exact_born/exact_born_L2_beta3p0_trajectories.csv",
           BASE/"data/exact_born_full/manifest.toml"]
    raw=[p.read_bytes() for p in paths]
    manifest=tomllib.loads(raw[2].decode())
    assert hashlib.sha256(raw[0]).hexdigest()==manifest["ground_orbital_sha256"]
    P=np.frombuffer(raw[0],dtype=np.float64).reshape((8,4),order="F")
    assert np.linalg.norm(P.T@P-np.eye(4))<1e-12
    F=list(itertools.combinations(range(8),4))
    pf=np.array([abs(np.linalg.det(P[list(f)]))**2 for f in F])
    assert abs(pf.sum()-1)<1e-12
    rows=list(csv.DictReader(raw[1].decode().splitlines()))
    E=[tuple(int(x)-1 for x in r["E_indices"].split(";")) for r in rows]
    assert len(rows)==70 and set(E)==set(F)
    a,beta=4,3.
    logw=np.array([2*math.log(math.comb(a,j))-4*beta*j for j in range(a+1)])
    w=np.exp(logw-logw.max());w/=w.sum()
    # Conditional law P(E|j) = sum_F pG(F) 1{|E intersect F|=j}/C(a,j)^2.
    conditional=np.zeros((a+1,len(E)))
    for f,p in zip(F,pf):
        for i,e in enumerate(E):
            j=len(set(f).intersection(e))
            conditional[j,i]+=p/math.comb(a,j)**2
    assert np.max(abs(conditional.sum(axis=1)-1))<1e-12
    marginal=w@conditional
    recorded=np.array([float(r["probability"]) for r in rows])
    error=float(np.max(abs(marginal-recorded)))
    assert error<1e-12
    results={}
    for name in ("S_A","S_B","S_C","A_minus_B","D"):
        values=np.array([float(r[name]) for r in rows])
        means=conditional@values
        variances=np.sum(conditional*(values[None,:]-means[:,None])**2,axis=1)
        mu=float(w@means)
        within=float(w@variances)
        between=float(w@((means-mu)**2))
        plain=float(marginal@((values-mu)**2))
        assert abs(plain-within-between)<1e-14
        equal=(a+1)*float((w*w)@variances)
        neyman=float(np.sum(w*np.sqrt(variances))**2)
        allocation=w*np.sqrt(variances)
        allocation/=allocation.sum()
        results[name]=dict(mean=mu,conditional_mean=means.tolist(),conditional_variance=variances.tolist(),
                          variance_times_total_samples=dict(unstratified_iid=plain,
                              proportional_allocation=within,equal_allocation=equal,oracle_neyman_allocation=neyman),
                          between_strata_variance=between,
                          variance_reduction_ratio=dict(proportional=plain/within,equal=plain/equal,oracle_neyman=plain/neyman),
                          oracle_neyman_sample_fractions=allocation.tolist(),
                          j0_fraction_of_proportional_variance=float(w[0]*variances[0]/within))
    # For a common allocation across several observables, do not apply separate
    # per-observable optimal fractions to the same claimed sample budget.
    result=dict(status="PASS",L=2,Nsites=8,Nparticles=a,beta=beta,deterministic_FE_pairs=4900,
                sampling="No random samples. All 70 F and all 70 E were summed.",
                conditional_formula="P(F,E|j)=pG(F)/binom(a,j)^2 when |F intersect E|=j. j has exact probability w_j proportional to binom(a,j)^2 exp(-4 beta j).",
                j_probabilities=w.tolist(),marginal_probability_max_error=error,observables=results,
                variance_formula="With fixed positive integer n_j and independent samples within each stratum: Var(sum_j w_j mean_j)=sum_j w_j^2 sigma_j^2/n_j. Numbers report the asymptotic n*Var assuming equal per-sample cost and ignore integer rounding/pilot costs.",
                limitations="Neyman allocations use exact variances only available in this small enumeration and are benchmarks, not a deployable zero-cost allocation. Large-system results and variation within j=0 are not settled. Separate optimal allocations cannot simultaneously be claimed for every observable at a single shared budget.",
                files={p.relative_to(BASE).as_posix():hashlib.sha256(b).hexdigest() for p,b in zip(paths,raw)})
    result["files"]["analysis/stratified_j_deterministic.py"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for p,b in zip(paths,raw):assert p.read_bytes()==b
    (HERE/"stratified_j_deterministic.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(status="PASS",j_probabilities=w.tolist(),marginal_probability_max_error=error,
                         results={name:r for name,r in results.items() if name in ("S_A","D")}),indent=2))


if __name__=="__main__":main()
