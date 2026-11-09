"""Deterministic small-system check of a strong-filter expansion; no MC.

The full-support formula is exact as a fixed-N asymptotic statement only if
every Fock minor is nonzero. Numerical rank detection below is evidence for
the saved L=3 state, not an algebraic proof of every minor's exact zero.
"""
from pathlib import Path
from itertools import combinations
import csv
import hashlib
import json
import math
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def f(x):
    return -math.log(x*x + (1-x)*(1-x))


def main():
    qpath = ROOT / "data/born_scaling/smallcheck/models/state_L3_staggered_mu1.0e-8_Q.bin"
    assert sha(qpath) == "2907b68467db033bbe94cb6bad8c9a2320a38973ace2fd903ef281efbd1d2a0d"
    Q = np.fromfile(qpath, dtype=np.float64).reshape((18, 9), order="F")
    assert np.max(abs(Q.T@Q-np.eye(9))) < 1e-12
    G = Q@Q.T
    sites = np.arange(18)
    regions = {
        "S_A": [2*(x+3*y)+s for y in range(3) for x in range(1) for s in (0,1)],
        "S_B": [2*(x+3*y)+s for y in range(2) for x in range(1) for s in (0,1)],
        "S_C": [2*(x+3*y)+s for y in range(1) for x in range(3) for s in (0,1)],
    }
    naive, correction = {}, {k:0. for k in regions}
    for name, R in regions.items():
        C = G[np.ix_(R, R)]
        mean_n = float(np.trace(C)); number_var = float(np.trace(C-C@C))
        assert abs(mean_n-len(R)/2) < 1e-12
        naive[name] = len(R)*(18-len(R)) + 4*number_var
    deficiency = {str(k):0 for k in range(10)}
    threshold_counts = {str(t):0 for t in (1e-13,1e-12,1e-11)}
    positive_min = math.inf
    zero_max = 0.
    W1_total = 0.
    dark_rows = []
    minsv_hist = {"under_1e-13":0,"1e-13_to_1e-11":0,"1e-11_to_1e-8":0,"above_1e-8":0}
    for S in combinations(range(18), 9):
        S = np.asarray(S)
        E = np.setdiff1d(sites, S)
        U, sv, Vh = np.linalg.svd(Q[S,:], full_matrices=True)
        for t in (1e-13,1e-12,1e-11):
            threshold_counts[str(t)] += int(sv[-1] < t)
        histkey = "under_1e-13" if sv[-1]<1e-13 else "1e-13_to_1e-11" if sv[-1]<1e-11 else "1e-11_to_1e-8" if sv[-1]<1e-8 else "above_1e-8"
        minsv_hist[histkey] += 1
        d = int(np.sum(sv < 1e-12)); deficiency[str(d)] += 1
        if d < 9:
            positive_min = min(positive_min, float(sv[-d-1] if d else sv[-1]))
        if d:
            zero_max = max(zero_max, float(sv[-d]))
        if d != 1:
            continue
        # Rank-one-deficient leading state: all S orbitals except one hole u,
        # and one particle v on E. The restricted S/E blocks are disjoint.
        hole = U[:,-1]
        particle = Q[E,:] @ Vh[-1,:]
        assert abs(float(particle@particle)-1) < 1e-10
        particle /= np.linalg.norm(particle)
        W1 = float(np.prod(sv[:-1]**2)); W1_total += W1
        row = dict(S_indices=";".join(str(i+1) for i in S),W1=W1)
        for name, R in regions.items():
            ph = float(np.sum(hole[np.isin(S,R)]**2))
            pe = float(np.sum(particle[np.isin(E,R)]**2))
            entropy = f(ph)+f(pe)
            cross = ph*(1-pe)+(1-ph)*pe
            delta = W1*(entropy-2*cross)
            correction[name] += delta
            row.update({f"{name}_hole_in_R":ph, f"{name}_particle_in_R":pe,
                        f"{name}_leading_entropy":entropy, f"{name}_coefficient_correction":delta})
        dark_rows.append(row)
    corrected = {k:naive[k]+correction[k] for k in regions}
    for obj in (naive, correction, corrected):
        obj["D"] = (2*obj["S_A"]+obj["S_C"])/3-obj["S_B"]
    exact_path = ROOT / "data/born_scaling/exact_L3_mu1e-8.csv"
    with exact_path.open(newline="") as stream:
        exact = {r["observable"]:float(r["mean"]) for r in csv.DictReader(stream) if float(r["beta"])==3.}
    q = math.exp(-12)
    comparison = [dict(observable=k, full_support_coefficient=naive[k],
                       rank_one_dark_correction=correction[k], corrected_coefficient=corrected[k],
                       exact_mean_beta3=exact[k], exact_mean_div_q_beta3=exact[k]/q,
                       naive_mean_prediction_beta3=q*naive[k], corrected_mean_prediction_beta3=q*corrected[k])
                  for k in ("S_A","S_B","S_C","D")]
    out = ROOT / "analysis/born_scaling_strong_measurement.csv"
    with out.open("w", newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(comparison[0]));writer.writeheader();writer.writerows(comparison)
    dark_path = ROOT / "analysis/born_scaling_strong_measurement_dark_outputs.csv"
    with dark_path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dark_rows[0]));writer.writeheader();writer.writerows(dark_rows)
    result = dict(status="PASS_NUMERICAL_RANK_DIAGNOSTIC", L=3, mu=1e-8, configurations=48620,
                  rank_threshold=1e-12, deficiency_counts=deficiency, zero_minor_counts_by_threshold=threshold_counts,
                  largest_classified_zero_singular_value=zero_max, smallest_classified_positive_singular_value=positive_min,
                  minimum_singular_value_histogram=minsv_hist, rank_one_dark_W1_sum=W1_total,
                  comparisons=comparison,
                  derivation={
                      "full_support":"At fixed N, if every Slater minor a_S is nonzero, mean S2(R)=2q E_F[N_cross(F)]+o(q). If <N_R>=r/2 this equals q r(N-r)+4q Var(N_R)+o(q). q=exp(-4 beta).",
                      "bright_output":"For a_S!=0, weighted entropy w_S S2=2q sum_{one-swap F crossing R}|a_F|^2+o(q); partition normalization Za=1+O(q) changes no leading O(q) coefficient.",
                      "rank_one_dark_output":"If rank(Q_S)=a-1, w_S=q W1+O(q^2). The leading normalized state fills S except a hole u and adds a particle v on E. S2(chi)=f(p_h)+f(p_e), f(p)=-ln[p^2+(1-p)^2].",
                      "dark_correction":"Its contribution replaces 2q W1[p_h(1-p_e)+(1-p_h)p_e] by q W1[f(p_h)+f(p_e)]. Rank deficiency>=2 contributes only at O(q^2) or higher.",
                      "D_full_support_leading_volume_term":"For r_A=r_C=N/3,r_B=2N/9, coefficient of q in mean D has geometric term 4N^2/81=16L^4/81 plus 4[(2 Var(N_A)+Var(N_C))/3-Var(N_B)].",
                  },
                  limitations=["Numerical rank tests at three thresholds are not an algebraic proof of exact zero minors.",
                               "A tiny nonzero minor can give a nonuniform q crossover; no uniform large-L remainder estimate is claimed.",
                               "The full-support formula must not be stated as universal for the current symmetry-constrained Slater state.",
                               "This fixed-size expansion does not predict an asymptotic log-L corner coefficient or replace production MC."],
                  files={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in (qpath,exact_path,out,dark_path,Path(__file__))})
    (ROOT/"analysis/born_scaling_strong_measurement.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({k:result[k] for k in ("status","deficiency_counts","smallest_classified_positive_singular_value","comparisons")},indent=2))


if __name__=="__main__":
    main()
