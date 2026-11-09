"""Independent exhaustive fixed-half Born benchmark, including latent-j strata.

Reads the model exported by export_exact_models.jl. No sampler/entropy routine
from the Monte Carlo implementation is used. E is zero-based in NPZ files.
L2 uses p=1/2 as a finite-system control; L3 uses the production p=1/3 geometry.
All entropies are natural-log Renyi-2. No small-probability stratum is discarded.
"""
from pathlib import Path
from itertools import combinations
import csv
import hashlib
import json
import math
import os
import time
import tomllib
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "exact_strata"
BETAS = (3.0, 4.0, 5.0)
NAMES = ("SA", "SB", "SC", "D", "VA", "VB", "VC", "DV",
         "VA_full", "VB_full", "VC_full", "DV_full")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rect(L, w, h, tx=0, ty=0):
    return np.array([2*((x+tx)%L+L*((y+ty)%L))+s
                     for y in range(h) for x in range(w) for s in (0, 1)])


def make_geometry(L, den):
    N = 2*L*L
    w = L//den
    p = 1/den
    origin = (rect(L, w, L), rect(L, w, L-w), rect(L, L, w))
    full = np.zeros((3, N, N))
    direct0 = np.zeros((3, N, N))
    for ty in range(L):
        for tx in range(L):
            regs = (rect(L, w, L, tx, ty), rect(L, w, L-w, tx, ty), rect(L, L, w, tx, ty))
            for k, reg in enumerate(regs):
                mask = np.zeros(N); mask[reg] = 1
                boundary = mask[:, None]*(1-mask)[None, :]
                full[k] += boundary/(L*L)
                if tx == ty == 0:
                    direct0[k] = boundary
    cells = np.arange(N)//2
    x, y = cells % L, cells//L
    dx, dy = (x[:, None]-x[None, :]) % L, (y[:, None]-y[None, :]) % L
    hx, hy = np.minimum(np.minimum(dx, L-dx), w)/L, np.minimum(np.minimum(dy, L-dy), w)/L
    kernel = hx*hy
    paired = (1-p)*full[0]+p*full[2]-full[1]
    err = float(np.max(abs(kernel-paired)))
    assert err < 1e-14
    return origin, direct0, full, kernel, err


def count_polynomial(Q, E):
    # Poisson-binomial positive recurrence, with only outside-[0,1] roundoff clipped.
    values = np.linalg.eigvalsh(Q[E].T@Q[E])
    excursion = max(0., float(-values.min()), float(values.max()-1))
    values = np.clip(values, 0, 1)
    c = np.array([1.])
    for value in values:
        new = np.zeros(len(c)+1)
        new[:-1] += (1-value)*c
        new[1:] += value*c
        c = new
    # At strong beta, an O(eps) spurious c0 on a zero-probability output can
    # dominate the tiny j=0 conditional variance. Cauchy-Binet gives stable
    # endpoint determinants, independently of eigenvalue subtraction near 1.
    complement = np.setdiff1d(np.arange(len(Q)), E, assume_unique=True)
    c[0] = np.linalg.det(Q[complement])**2
    c[-1] = np.linalg.det(Q[E])**2
    return c, excursion


def moments(prob, values):
    mean = prob@values
    centered = values-mean
    var = prob@(centered*centered)
    fourth = prob@(centered**4)
    kurt = np.divide(fourth, var*var, out=np.zeros_like(var), where=var>1e-30)
    return mean, var, kurt


def one_size(L):
    started = time.perf_counter()
    model_file = OUT / "models" / f"model_L{L}.toml"
    meta = tomllib.loads(model_file.read_text())
    qfile = model_file.parent/meta["Q_file"]
    assert sha(qfile) == meta["Q_sha256"]
    N, a = meta["N"], meta["n"]
    Q = np.fromfile(qfile, dtype="<f8").reshape(N, a, order="F")
    Q = np.linalg.qr(Q)[0]  # removes export-level orthogonality roundoff, same occupied space
    G0 = Q@Q.T
    den = 2 if L == 2 else 3
    p = 1/den
    origin, K0, Kfull, KD, geometry_error = make_geometry(L, den)
    V0 = np.einsum("kij,ij->k", K0, G0*G0)
    DV0 = float(np.sum(KD*G0*G0))
    initial_s = np.array([-np.log(v*v+(1-v)**2).sum()
                          for R in origin for v in [np.linalg.eigvalsh(G0[np.ix_(R,R)])]])
    configs = np.array(list(combinations(range(N), a)), dtype=np.int16)
    coeff = np.empty((len(configs), a+1))
    excursion = 0.
    for i, E in enumerate(configs):
        coeff[i], err = count_polynomial(Q, E)
        excursion = max(excursion, err)
    normalizers = np.array([math.comb(a,j)**2 for j in range(a+1)], dtype=float)
    column_errors = coeff.sum(axis=0)/normalizers-1
    assert max(abs(column_errors)) < 1e-11
    conditionals = coeff/normalizers
    # Normalize the floating-point conditional probabilities only after recording the identity error.
    conditionals /= conditionals.sum(axis=0)
    double_enum_error = None
    if L == 2:
        Fprob = np.array([np.linalg.det(Q[F])**2 for F in configs])
        direct_coeff = np.zeros_like(coeff)
        for ei, E in enumerate(configs):
            overlap = np.isin(configs, E).sum(axis=1)
            direct_coeff[ei] = np.bincount(overlap, weights=Fprob, minlength=a+1)
        double_enum_error = float(np.max(abs(direct_coeff-coeff)))
        assert double_enum_error < 1e-12
    np.savez_compressed(OUT/f"L{L}_conditional_probabilities.npz", E=configs,
                        c=coeff, p_E_given_j=conditionals, Q=Q, G0=G0, kernel_DV=KD)
    summary = dict(L=L, N=N, a=a, mu=1e-8, geometry_p=p,
                   geometry_note="L2 control p=1/2" if L==2 else "production p=1/3",
                   number_of_outputs=len(configs), model=meta,
                   original_region_indices_1based=[(R+1).tolist() for R in origin],
                   column_normalization_relative_errors=column_errors.tolist(),
                   eigenvalue_max_outside_unit_interval=excursion,
                   L2_all_F_E_coefficient_max_error=double_enum_error,
                   geometry_pairwise_kernel_error=geometry_error,
                   initial_S2=initial_s.tolist(), initial_V=V0.tolist(), initial_DV_full=DV0,
                   cases=[])
    rows=[]
    for beta in BETAS:
        t = math.exp(-2*beta); q = t*t
        jvals = np.arange(a+1)
        term = normalizers*np.exp(-4*beta*jvals)
        Z = float(term.sum()); pj=term/Z
        Za1 = math.fsum(math.comb(a-1,j)**2*q**j for j in range(a))
        lam = 1-(1-t)**2*Za1/Z
        values = np.empty((len(configs), len(NAMES)))
        logweights = np.empty(len(configs))
        nmeans = np.empty((len(configs),3))
        meanG = np.zeros((N,N)); secondG = np.zeros((N,N))
        for i, E in enumerate(configs):
            weighted = Q.copy(); weighted[E] *= t
            P, Rqr = np.linalg.qr(weighted)
            G = P@P.T
            logweights[i] = 2*np.log(abs(np.diag(Rqr))).sum()
            weight = math.exp(logweights[i])/Z
            meanG += weight*G; secondG += weight*(G*G)
            s = np.array([-np.log(ev*ev+(1-ev)**2).sum()
                          for R in origin for ev in [np.clip(np.linalg.eigvalsh(G[np.ix_(R,R)]),0,1)]])
            v = np.einsum("kij,ij->k", K0, G*G)
            vf = np.einsum("kij,ij->k", Kfull, G*G)
            d = (1-p)*s[0]+p*s[2]-s[1]
            dv = (1-p)*v[0]+p*v[2]-v[1]
            dvf = float(np.sum(KD*G*G))
            values[i] = np.r_[s,d,v,dv,vf,dvf]
            nmeans[i] = [G.diagonal()[R].sum() for R in origin]
        directprob = np.exp(logweights)/Z
        sumprob = float(directprob.sum())
        assert abs(sumprob-1) < 1e-11
        directprob /= sumprob; meanG /= sumprob; secondG /= sumprob
        mixtureprob = conditionals@pj
        mean, variance, kurtosis = moments(directprob, values)
        cm = conditionals.T@values
        cv = np.array([conditionals[:,j]@((values-cm[j])**2) for j in range(a+1)])
        ck = np.array([moments(conditionals[:,j], values)[2] for j in range(a+1)])
        within = pj@cv
        between = pj@((cm-mean)**2)
        neyman = (pj@np.sqrt(cv))**2
        equal = (a+1)*((pj*pj)@cv)
        mean_prediction = lam*G0+(1-lam)*np.diag(np.diag(G0))
        channel_error = float(np.max(abs(meanG-mean_prediction)))
        mixture_mean_error = float(np.max(abs(pj@cm-mean)))
        mixture_prob_error = float(np.max(abs(mixtureprob-directprob)))
        variance_identity_error = float(np.max(abs(variance-within-between)))
        total_var_error = float(np.max(abs(mean[4:7]+directprob@((nmeans-directprob@nmeans)**2)-V0)))
        rms_slack = secondG-lam*lam*(G0*G0)
        assert channel_error < 1e-10
        assert mixture_mean_error < 1e-10
        assert mixture_prob_error < 1e-11
        assert variance_identity_error < 1e-10
        assert total_var_error < 1e-10
        assert rms_slack.min() > -1e-11
        assert mean[-1] >= lam*lam*DV0-1e-11
        assert lam >= 1/math.cosh(2*beta)-1e-14
        assert np.all(mean[:3] >= 2*lam*lam*V0-1e-11)
        assert np.all(mean[:3] <= 4*math.log(2)*V0+1e-11)
        assert np.all(4*math.log(2)*V0 <= 2*math.log(2)*initial_s+1e-11)
        case=dict(beta=beta, t=t, lambda1=lam, stratum_probabilities=pj.tolist(),
            mean=dict(zip(NAMES,mean.tolist())), variance=dict(zip(NAMES,variance.tolist())),
            standardized_fourth_moment=dict(zip(NAMES,kurtosis.tolist())),
            per_draw_proportional_allocation_variance=dict(zip(NAMES,within.tolist())),
            per_draw_neyman_allocation_variance=dict(zip(NAMES,neyman.tolist())),
            per_draw_equal_allocation_variance=dict(zip(NAMES,equal.tolist())),
            proportional_variance_ratio=dict(zip(NAMES,(within/variance).tolist())),
            neyman_variance_ratio=dict(zip(NAMES,(neyman/variance).tolist())),
            equal_allocation_variance_ratio=dict(zip(NAMES,(equal/variance).tolist())),
            mean_G_formula_max_error=channel_error,
            min_second_moment_lower_bound_slack=float(rms_slack.min()),
            DV_full_lower_bound=lam*lam*DV0,
            DV_full_lower_bound_ratio=float(mean[-1]/(lam*lam*DV0)),
            entropy_lower_bound_2_lambda_sq_V0=(2*lam*lam*V0).tolist(),
            universal_finite_beta_lambda_lower_bound=1/math.cosh(2*beta),
            entropy_order_lower_bound=(initial_s/(2*math.log(2)*math.cosh(2*beta)**2)).tolist(),
            area_bound_4ln2_V0=(4*math.log(2)*V0).tolist(),
            checks=dict(direct_QR_probability_sum_error=abs(sumprob-1),
                        mixture_probability_max_error=mixture_prob_error,
                        mixture_mean_max_error=mixture_mean_error,
                        total_variance_decomposition_error=variance_identity_error,
                        posterior_number_total_variance_error=total_var_error))
        for j in range(a+1):
            for k,name in enumerate(NAMES):
                rows.append(dict(L=L,p=p,beta=beta,j=j,observable=name,pj=pj[j],
                    conditional_mean=cm[j,k], conditional_variance=cv[j,k],
                    conditional_standardized_fourth_moment=ck[j,k],
                    contribution_to_mean=pj[j]*cm[j,k],
                    contribution_to_within_variance=pj[j]*cv[j,k],
                    contribution_to_neyman_sum=pj[j]*math.sqrt(cv[j,k])))
        np.savez_compressed(OUT/f"L{L}_beta{beta:g}.npz", E=configs, names=np.array(NAMES),
                            observables=values, probabilities=directprob,
                            logweights=logweights, pj=pj, conditional_mean=cm,
                            conditional_variance=cv, meanG=meanG, mean_absG2=secondG,
                            number_means=nmeans)
        summary["cases"].append(case)
        print(f"L={L} beta={beta:g} exact D={mean[3]:.10g} SA={mean[0]:.10g} "
              f"DVfull={mean[-1]:.10g} channelerr={channel_error:.3g} "
              f"D variance ratios prop/Neyman={within[3]/variance[3]:.5g}/{neyman[3]/variance[3]:.5g}",flush=True)
    summary["runtime_seconds"]=time.perf_counter()-started
    (OUT/f"L{L}_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    with (OUT/f"L{L}_strata.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return summary


def main():
    OUT.mkdir(exist_ok=True)
    results=[one_size(L) for L in (2,3)]
    receipt=dict(status="PASS", script_sha256=sha(__file__), numpy_version=np.__version__,
        betas=list(BETAS), results=results,
        interpretation="Allocation ratios are asymptotic variance per fixed TOTAL independent draws at equal per-draw cost; not measured wall-time gains. Equal j allocation can be worse than IID. Conditional fourth moments reveal residual within-stratum tails.",
        files={str(path.relative_to(OUT)):sha(path) for path in sorted(OUT.rglob("*"))
               if path.is_file() and path.name!="receipt.json"})
    (OUT/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print("PASS: all outputs enumerated; all polynomial, channel, variance and bound checks passed.",flush=True)


if __name__=="__main__": main()
