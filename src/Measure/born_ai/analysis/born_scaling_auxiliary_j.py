"""Exact auxiliary-j moments of the fixed-half normalizer; no Monte Carlo.

j is an auxiliary overlap |E intersect F|. This script does not assign an
excitation count, entropy, physical correlation length or corner coefficient.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import sys
try:
    import mpmath as mp
except ImportError:
    # Reuse the existing audited dependency; do not install/modify packages.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ai/analysis_beta_patterns/audit_dependencies"))
    import mpmath as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
SIZES = tuple(range(6, 25, 3))
BETAS = tuple(k / 2 for k in range(7))
mp.mp.dps = 80


def exact_moments(a, beta):
    # The Float64 result uses logarithmic recurrence and normalization.
    j = np.arange(a + 1, dtype=float)
    logw = np.zeros(a + 1)
    for k in range(1, a + 1):
        logw[k] = logw[k - 1] + 2 * math.log((a - k + 1) / k) - 4 * beta
    peak = max(logw)
    logz = peak + math.log(math.fsum(np.exp(logw - peak)))
    p = np.exp(logw - logz)
    mean = float(np.dot(j, p))
    var = float(np.dot((j - mean) ** 2, p))

    # Independent high-precision reference: exact binomial integers, not the
    # Float64 recurrence. All 49 finite sums are computed, with no sampling.
    q = mp.exp(-4 * mp.mpf(str(beta)))
    weights = [mp.mpf(math.comb(a, k)) ** 2 * q ** k for k in range(a + 1)]
    total = mp.fsum(weights)
    mpmean = mp.fsum(k * w for k, w in enumerate(weights)) / total
    mpvar = mp.fsum((k - mpmean) ** 2 * w for k, w in enumerate(weights)) / total
    mpgt0 = 1 - 1 / total
    errors = dict(log_normalizer=abs(logz - float(mp.log(total))),
        mean=abs(mean - float(mpmean)), variance=abs(var - float(mpvar)),
        probability_positive=abs(-math.expm1(-logz) - float(mpgt0)),
        probability_sum=abs(math.fsum(p) - 1))
    assert max(errors.values()) < 1e-8
    if beta == 0:
        assert abs(mean - a / 2) < 1e-9
        assert abs(var - a * a / (4 * (2 * a - 1))) < 1e-9
        assert mp.almosteq(total, mp.mpf(math.comb(2*a,a)), rel_eps=mp.mpf("1e-70"))
    z = a * math.exp(-2 * beta)
    # Bounds from 1+z^2 <= Z <= I0(2z); valid for every finite a and beta.
    log_bessel = float(mp.log(mp.besseli(0, 2 * mp.mpf(str(z)))))
    assert math.log1p(z * z) <= logz + 1e-10
    assert logz <= log_bessel + 1e-10
    cdf = np.cumsum(p)
    return dict(log_Z=logz,log10_probability_j0=-logz/math.log(10),
        probability_j0=float(mp.mpf(1)/total),
        probability_j0_high_precision=mp.nstr(1/total, 24),
        probability_j_positive=-math.expm1(-logz),
        mean_j=mean,variance_j=var,mean_j_over_a=mean/a,
        mode_j=int(np.argmax(p)),quantile_j_05=int(np.searchsorted(cdf,.05)),
        quantile_j_95=int(np.searchsorted(cdf,.95)),quantile_j_99=int(np.searchsorted(cdf,.99)),
        q=math.exp(-4*beta),sqrt_q=math.exp(-2*beta),z=a*math.exp(-2*beta),
        exact_weight_ratio_j1_j0=z*z,
        probability_positive_lower_bound=z*z/(1+z*z),
        probability_positive_bessel_upper_bound=-math.expm1(-log_bessel),
        full_local_probability_fixed_half=math.exp(logz-2*a*math.log1p(math.exp(-2*beta)))),errors


def main():
    rows = []
    maxima = {}
    for L in SIZES:
        for beta in BETAS:
            values, errors = exact_moments(L * L, beta)
            rows.append(dict(L=L,a=L*L,beta=beta,**values))
            for key, value in errors.items():
                maxima[key] = max(maxima.get(key, 0.), value)
    csvpath = OUT / "born_scaling_auxiliary_j.csv"
    with csvpath.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    reference, _ = exact_moments(9, 3.)
    payload = {
        "status": "PASS", "rows": len(rows), "sizes": SIZES, "betas": BETAS,
        "precision_reference_decimal_digits": 80,
        "maximum_float64_vs_exact_binomial_sum_errors": maxima,
        "definition": "p(j)=binom(a,j)^2 exp(-4 beta j)/Z_a; a=L^2; j=|E intersect F| in the auxiliary joint distribution. p(F,E)=p_G(F) exp(-4 beta |E intersect F|)/Z_a and j is independent of F.",
        "exact_identities": [
            "p(j+1)/p(j)=((a-j)/(j+1))^2 exp(-4 beta)",
            "z=a exp(-2 beta), so p(1)/p(0)=z^2 exactly",
            "P(j>0)=1-1/Z_a; E[j]=-(1/4) d log(Z_a)/d beta; Var(j)=(1/16) d^2 log(Z_a)/d beta^2",
            "1+z^2 <= Z_a <= I_0(2z), using binom(a,j)<=a^j/j!",
            "z^2/(1+z^2) <= P(j>0) <= 1-1/I_0(2z)",
            "At beta=0: Z=binom(2a,a), E[j]=a/2, Var(j)=a^2/[4(2a-1)]",
            "Complete-local-POVM P(|E|=a)=Z_a/(1+exp(-2 beta))^(2a); this is separate from conditional p(j).",
        ],
        "asymptotics_with_conditions": [
            "For z<<1, P(j>0)=z^2+O(z^4), E[j]=z^2+O(z^4).",
            "Double scaling a->infinity, exp(-2 beta)=z/a with fixed z gives p(j)->z^(2j)/[(j!)^2 I_0(2z)] and E[j]->z I_1(2z)/I_0(2z). This is not a Poisson distribution.",
            "For fixed finite beta and a->infinity, saddle point j/a->1/(1+exp(2 beta)); the leading Gaussian variance is a*t/[2(1+t)^2], t=exp(-2 beta).",
            "z around one corresponds to beta around log(L), or L around exp(beta); this locates an auxiliary finite-size normalization crossover only.",
        ],
        "limitations": [
            "j is not a measured trajectory excitation, not a label of a physical mode, and not a geometric configuration class.",
            "The auxiliary F is not physically measured. The postmeasurement entropy must always be calculated from M_E applied to the original Slater state.",
            "These moments are independent of the detailed Slater projector and therefore cannot determine entropy, a physical correlation length, a phase transition, or c(beta).",
            "Small probability j>0 does not bound its entropy contribution, and j=0 can still have rare F-dependent entropy contributions.",
            "The grid is a deterministic finite-sum calculation, not additional Monte Carlo data or an entropy extrapolation.",
            "Float64 probability_j0 can underflow to zero; log10_probability_j0 and the high-precision string preserve its strictly positive value.",
        ],
        "L3_beta3_reference": dict(L=3,a=9,beta=3.,**reference),
        "files": {
            "analysis/"+csvpath.name: hashlib.sha256(csvpath.read_bytes()).hexdigest(),
            "analysis/"+Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    out = OUT / "born_scaling_auxiliary_j.json"
    out.write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    print("PASS:", len(rows), "exact finite sums; error maxima", maxima)
    for row in rows:
        if row["beta"] == 3:
            print({k:row[k] for k in ("L","beta","z","probability_j_positive","mean_j")})


if __name__ == "__main__":
    main()
