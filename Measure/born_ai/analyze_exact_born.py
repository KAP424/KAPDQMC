"""Compare exact fixed-half Born means with local and IID Monte Carlo.

This analysis describes finite-system entropy averages, never scaling fits.
Run only after compare_exact_born.jl has completed its immutable run manifest.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import hashlib
import json
import math
import os
import tomllib

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "exact_born"
OUT = ROOT / "analysis" / "exact_born"
FIG = ROOT / "figures"
BETAS = (0.0, 0.5, 1.0, 2.0, 3.0)
SIZES = (2, 3)
OBSERVABLES = ("S_A", "S_B", "S_C", "A_minus_B", "D")
PLOT_OBSERVABLES = ("S_A", "S_B", "A_minus_B", "D")
LABELS = {"S_A": r"$\langle S_2(A)\rangle$", "S_B": r"$\langle S_2(B)\rangle$",
          "S_C": r"$\langle S_2(C)\rangle$", "A_minus_B": r"$\langle S_2(A)-S_2(B)\rangle$",
          "D": r"$\langle(2S_2(A)+S_2(C))/3-S_2(B)\rangle$"}
STYLES = {"local": dict(color="#286B9C", marker="o", ms=9, mfc="white", mew=1.6),
          "iid": dict(color="#D47728", marker="s", ms=4.5, mfc="#D47728", mew=1),
          "iid_oracle": dict(color="#718535",marker="D",ms=6,mfc="white",mew=1.3)}
FILES = {}


def tracked(path):
    path = Path(path).resolve()
    relative = Path(os.path.relpath(path, ROOT)).as_posix()
    contents = path.read_bytes()
    FILES[relative] = hashlib.sha256(contents).hexdigest()
    return contents


def read_csv(path):
    return list(csv.DictReader(tracked(path).decode("utf-8-sig").splitlines()))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tracked(path)


def verify_manifest(path):
    payload = tracked(path).decode("utf-8-sig")
    manifest = json.loads(payload) if path.suffix == ".json" else tomllib.loads(payload)
    assert manifest["status"] in ("PASS", "complete"), "Run is incomplete"
    for relative, digest in manifest["files"].items():
        actual = hashlib.sha256(tracked(ROOT / relative)).hexdigest()
        assert actual == digest, f"Source SHA changed: {relative}"
    for source, digest in manifest["source_sha256"].items():
        actual = hashlib.sha256(tracked(Path(source))).hexdigest()
        assert actual == digest, f"Executed source changed: {source}"
    assert manifest["ensemble"] == "fixed_half" and manifest["block_size"] == 100
    seeds = [run["seed"] for run in manifest["runs"]]
    assert len(seeds) == 40 and len(set(seeds)) == len(seeds), "Sampler/chain seeds are not disjoint"
    return manifest


def independent_statistics(chains, block_size):
    n = sum(len(x) for x in chains)
    values = np.concatenate(chains)
    if np.ptp(values) < 1e-12:
        return float(values.mean()), 0.0, float(n)
    blocks = np.concatenate([x.reshape(-1, block_size).mean(axis=1) for x in chains])
    denominator = 0.0
    for x in chains:
        if np.ptp(x) < 1e-12:
            g = 1.0
        else:
            centered = x - x.mean()
            # FFT independently reconstructs the same finite autocovariances;
            # the experiment computes them with explicit Julia dot products.
            fft_size = 1 << (2*len(x)-1).bit_length()
            spectrum = np.fft.rfft(centered, n=fft_size)
            covariance = np.fft.irfft(spectrum*spectrum.conj(), n=fft_size)
            maxlag = min(2000, len(x)//2)
            rho = covariance[:maxlag+1]/covariance[0]
            total, previous = 0.0, math.inf
            for lag in range(0, len(rho)-1, 2):
                pair = float(rho[lag]+rho[lag+1])
                if pair <= 0:
                    break
                pair = min(pair, previous)
                total += pair
                previous = pair
            g = max(1.0, -1+2*total)
        denominator += len(x)*g
    return float(values.mean()), float(blocks.std(ddof=1)/math.sqrt(len(blocks))), n*n/denominator


def audit_raw(table):
    errors = dict(exact_probability_sum=0.0, exact_probability=0.0, exact_mean=0.0,
                  MC_mean=0.0, block_SE=0.0, ESS=0.0, trajectory_entropy=0.0)
    counts = dict(exact_trajectories=0, MC_samples=0, independent_statistics=0)
    case_rows = []
    exact_moments = {}
    for L in SIZES:
        nsites, particles = 2*L*L, L*L
        for beta in BETAS:
            tag = str(beta).replace(".", "p")
            prefix = DATA / f"exact_born_L{L}_beta{tag}"
            exact_rows = read_csv(Path(str(prefix)+"_trajectories.csv"))
            assert len(exact_rows) == math.comb(nsites, particles)
            exact_values = {}
            p, log_weight, entropies = [], [], []
            for row in exact_rows:
                assert int(row["L"]) == L and float(row["beta"]) == beta and row["ensemble"] == "fixed_half"
                key = row["E_indices"]
                indices = list(map(int, key.split(";")))
                assert indices == sorted(set(indices)) and len(indices) == particles
                assert min(indices) >= 1 and max(indices) <= nsites and key not in exact_values
                values = np.array([float(row[obs]) for obs in OBSERVABLES])
                assert abs(values[3]-(values[0]-values[1])) < 1e-12
                assert abs(values[4]-((2*values[0]+values[2])/3-values[1])) < 1e-12
                exact_values[key] = values
                p.append(float(row["probability"]))
                log_weight.append(float(row["log_weight"]))
                entropies.append(values)
            p = np.array(p)
            logs = np.array([2*(math.lgamma(particles+1)-math.lgamma(j+1)-math.lgamma(particles-j+1))-4*beta*j
                             for j in range(particles+1)])
            logZ = float(logs.max()+np.log(np.exp(logs-logs.max()).sum()))
            errors["exact_probability_sum"] = max(errors["exact_probability_sum"], abs(float(p.sum())-1))
            errors["exact_probability"] = max(errors["exact_probability"], float(np.max(abs(p-np.exp(np.array(log_weight)-logZ)))))
            assert np.min(p) >= 0 and abs(float(p.sum())-1) < 2e-10
            assert np.max(abs(p-np.exp(np.array(log_weight)-logZ))) < 2e-10
            entropy_array = np.array(entropies)
            exact_means = np.sum(p[:,None]*entropy_array, axis=0)
            exact_variances = np.sum(p[:,None]*(entropy_array-exact_means)**2,axis=0)
            for i, obs in enumerate(OBSERVABLES):
                err = abs(exact_means[i]-table[L,beta,"exact",obs]["mean"])
                errors["exact_mean"] = max(errors["exact_mean"], float(err))
                assert err < 2e-10
                exact_moments[L,beta,obs] = dict(mean=float(exact_means[i]),
                    variance=0.0 if beta==0 else float(exact_variances[i]))
            counts["exact_trajectories"] += len(exact_rows)
            del exact_rows, entropies
            samples = read_csv(Path(str(prefix)+"_mc.csv"))
            assert len(samples) == 60000
            by_chain = defaultdict(list)
            for row in samples:
                assert int(row["L"]) == L and float(row["beta"]) == beta and row["ensemble"] == "fixed_half"
                key = row["method"], int(row["chain"])
                assert int(row["sample"]) == len(by_chain[key])+1
                values = np.array([float(row[obs]) for obs in OBSERVABLES])
                error = float(np.max(abs(values-exact_values[row["E_indices"]])))
                errors["trajectory_entropy"] = max(errors["trajectory_entropy"], error)
                assert error < 1e-9
                by_chain[key].append(values)
            assert set(by_chain) == {("local",1),("local",2),("local",3),("iid",1)}
            for key, values in by_chain.items():
                assert len(values) == (10000 if key[0] == "local" else 30000)
            arrays = {key: np.array(values) for key, values in by_chain.items()}
            for method in ("local", "iid"):
                chains = [values for (name, _), values in sorted(arrays.items()) if name == method]
                for i, obs in enumerate(OBSERVABLES):
                    row = table[L,beta,method,obs]
                    mu, se, ess = independent_statistics([x[:,i] for x in chains], row["block_size"])
                    for label, value, expected, tolerance in (("MC_mean",mu,row["mean"],2e-11),
                        ("block_SE",se,row["SE"],2e-11),("ESS",ess,row["ESS"],2e-7)):
                        errors[label] = max(errors[label], abs(value-expected))
                        assert abs(value-expected) < tolerance, (L,beta,method,obs,label,value,expected)
                    counts["independent_statistics"] += 1
            counts["MC_samples"] += len(samples)
            case_rows.append(dict(L=L,beta=beta,exact_configurations=len(exact_values),MC_samples=len(samples),
                                  local_chains=3,iid_chains=1,ensemble="fixed_half"))
            del samples, arrays, by_chain, exact_values
    return dict(status="PASS",maximum_absolute_errors=errors,counts=counts,cases=case_rows), exact_moments


def read_summary():
    raw = read_csv(DATA / "exact_born_summary.csv")
    table = {}
    for row in raw:
        assert row["ensemble"] == "fixed_half", "Do not mix full_local with the fixed-half comparison"
        key = (int(row["L"]), float(row["beta"]), row["method"], row["observable"])
        assert key not in table, f"Duplicate summary row: {key}"
        assert key[0] in SIZES and key[1] in BETAS
        assert key[2] in ("exact", "local", "iid") and key[3] in OBSERVABLES
        converted = dict(row)
        for name in ("mean", "SE", "ESS", "exact_mean", "difference"):
            converted[name] = float(row[name])
            assert math.isfinite(converted[name]), (key, name)
        for name in ("n", "chains", "block_size", "burnin_per_chain", "refresh_every"):
            converted[name] = int(row[name])
        assert converted["SE"] >= 0
        if key[2] == "exact":
            assert converted["n"] == math.comb(2 * key[0] ** 2, key[0] ** 2)
            assert converted["SE"] == 0
        else:
            assert converted["n"] == 30000, (key, converted["n"])
        table[key] = converted
    expected = {(L, beta, method, obs) for L in SIZES for beta in BETAS
                for method in ("exact", "local", "iid") for obs in OBSERVABLES}
    assert set(table) == expected, f"Missing rows: {expected-set(table)}"
    for L in SIZES:
        for beta in BETAS:
            for method in ("exact", "local", "iid"):
                values = {obs: table[L, beta, method, obs]["mean"] for obs in OBSERVABLES}
                assert abs(values["A_minus_B"] - values["S_A"] + values["S_B"]) < 2e-11
                assert abs(values["D"] - (2*values["S_A"]+values["S_C"])/3 + values["S_B"]) < 2e-11
            for obs in OBSERVABLES:
                exact = table[L, beta, "exact", obs]["mean"]
                for method in ("exact", "local", "iid"):
                    row = table[L, beta, method, obs]
                    assert abs(row["exact_mean"] - exact) < 2e-11
                    assert abs(row["difference"] - (row["mean"] - exact)) < 2e-11
    return table


def comparisons(table, moments):
    records = []
    for L in SIZES:
        for beta in BETAS:
            for obs in OBSERVABLES:
                exact = table[L, beta, "exact", obs]["mean"]
                for method in ("local", "iid"):
                    row = table[L, beta, method, obs]
                    difference = row["mean"] - exact
                    # beta=0 is independent of the sampled outcome; floating-point
                    # roundoff is not a statistical error bar or a defined z-score.
                    constant = beta == 0 or row["SE"] == 0
                    if constant:
                        assert abs(difference) < 2e-11, (L, beta, method, obs, difference)
                        z = None
                    else:
                        z = difference / row["SE"]
                        reported = float(row["z_score"])
                        assert abs(z-reported) < 1e-8*max(1, abs(z))
                    oracle_se=math.sqrt(moments[L,beta,obs]["variance"]/row["n"]) if method=="iid" else None
                    oracle_z=difference/oracle_se if oracle_se is not None and oracle_se>0 else None
                    records.append(dict(L=L, beta=beta, ensemble="fixed_half", observable=obs,
                        method=method, exact_mean=exact, MC_mean=row["mean"], MCSE=row["SE"],
                        difference=difference, standardized_residual=z,
                        residual_status="undefined_zero_variance_roundoff_only" if constant else "defined",
                        exact_population_variance=moments[L,beta,obs]["variance"],
                        oracle_iid_SE=oracle_se,oracle_iid_standardized_residual=oracle_z,
                        n=row["n"], chains=row["chains"], block_size=row["block_size"],
                        burnin_per_chain=row["burnin_per_chain"], ESS=row["ESS"],
                        quantity_role="finite_system_entropy_average_not_scaling_coefficient"))
    return records


def beta_table(table,moments):
    result=[]
    for L in SIZES:
        for beta in BETAS:
            for obs in OBSERVABLES:
                row=dict(L=L,beta=beta,ensemble="fixed_half",observable=obs,
                         exact_mean=table[L,beta,"exact",obs]["mean"],
                         exact_population_variance=moments[L,beta,obs]["variance"],
                         oracle_iid_SE=math.sqrt(moments[L,beta,obs]["variance"]/table[L,beta,"iid",obs]["n"]))
                for method in ("local","iid"):
                    source=table[L,beta,method,obs]
                    row[f"{method}_mean"]=source["mean"]
                    row[f"{method}_block_SE"]=source["SE"]
                    row[f"{method}_samples"]=source["n"]
                result.append(row)
    return result


def audit_tail_diagnostic(moments,records):
    diagnostic=json.loads(tracked(DATA/"exact_born_tail_diagnostics.json").decode())
    assert diagnostic["status"]=="complete" and diagnostic["audit_status"]=="PASS"
    for relative,digest in diagnostic["files"].items():
        assert hashlib.sha256(tracked(ROOT/relative)).hexdigest()==digest,relative
    iid={(r["L"],r["beta"],r["observable"]):r for r in records if r["method"]=="iid"}
    external=read_csv(DATA/"exact_born_exact_moments.csv")
    assert len(external)==50
    for row in external:
        key=int(row["L"]),float(row["beta"]),row["observable"]
        assert row["ensemble"]=="fixed_half"
        for column,value in (("exact_mean",moments[key]["mean"]),("exact_variance",moments[key]["variance"]),
                             ("oracle_iid_SE",iid[key]["oracle_iid_SE"])):
            assert abs(float(row[column])-value)<2e-11,(key,column)
    exact=read_csv(DATA/"exact_born_L2_beta3p0_trajectories.csv")
    samples=read_csv(DATA/"exact_born_L2_beta3p0_mc.csv")
    for tail in diagnostic["tails"]:
        obs=tail["observable"]
        threshold=tail["threshold"]
        selected=[r for r in exact if abs(float(r[obs]))>threshold]
        mass=sum(float(r["probability"]) for r in selected)
        assert len(selected)==tail["configurations"]
        assert abs(mass-tail["probability_mass"])<1e-12
        assert abs(math.exp(30000*math.log1p(-mass))-tail["iid_zero_hit_probability"])<2e-10
        for method in ("local","iid"):
            hits=sum(r["method"]==method and abs(float(r[obs]))>threshold for r in samples)
            assert hits==tail[f"{method}_observed_hits"]
    return dict(status="PASS",selection_policy=diagnostic["selection_policy"],tails=diagnostic["tails"],
        interpretation="Exact-distribution IID SE is a root mean square uncertainty, not a guarantee of Gaussian residuals or 95% coverage for rare-event sample means.")


def compare_reproduction_snapshot(manifest):
    path=OUT/"pre_reprofix_snapshot.json"
    if not path.exists():
        return dict(status="not_requested",reason="No earlier numerical snapshot is present.")
    baseline=json.loads(tracked(path).decode())
    maximum_error=0.0
    comparisons=0
    for table_name in ("comparison","beta_table"):
        current=read_csv(OUT/f"{table_name}.csv")
        previous=baseline[table_name]
        assert len(current)==len(previous)
        for old,new in zip(previous,current):
            assert old.keys()==new.keys()
            for field in old:
                try:
                    a,b=float(old[field]),float(new[field])
                except ValueError:
                    assert old[field]==new[field],(table_name,field,old[field],new[field])
                    continue
                error=abs(a-b)
                assert error<=5e-12+5e-10*max(abs(a),abs(b)),(table_name,field,a,b)
                maximum_error=max(maximum_error,error)
                comparisons+=1
    old_raw=baseline["old_raw_csv_sha256"]
    unchanged=[p for p,h in old_raw.items() if manifest["files"].get(p)==h]
    return dict(status="PASS",scope="Same seeds and sample counts after dependency-list-only reproducibility fixes.",
        compared_numeric_fields=comparisons,maximum_absolute_change=maximum_error,
        old_raw_csv_files=len(old_raw),byte_identical_raw_csv_files=len(unchanged),
        changed_raw_csv_files=[p for p in old_raw if p not in unchanged],
        old_analysis_sha256=baseline["old_analysis_sha256"])


def save(fig, stem):
    for suffix in ("png", "pdf"):
        path = FIG / f"{stem}.{suffix}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        tracked(path)
    plt.close(fig)


def base_figure(title):
    fig, axes = plt.subplots(2, 4, figsize=(15.8, 8.1), squeeze=False)
    fig.suptitle(title, fontsize=17, y=.985)
    for row, L in enumerate(SIZES):
        for col, obs in enumerate(PLOT_OBSERVABLES):
            ax = axes[row, col]
            ax.set_title(f"L = {L}: " + LABELS[obs], fontsize=12)
            ax.set_xlabel(r"Measurement strength $\beta$")
            ax.set_xticks(BETAS)
            ax.set_xlim(-.16, 3.16)
            ax.grid(axis="y", color="#dddddd", linewidth=.7)
            ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=.065, right=.995, top=.855, bottom=.21, hspace=.5, wspace=.33)
    return fig, axes


def make_plots(table, records):
    legend = [Line2D([], [], color="#242424", lw=1.6, label="Exact enumeration"),
              Line2D([], [], linestyle="none", label="Local swaps", **STYLES["local"]),
              Line2D([], [], linestyle="none", label="Independent IID", **STYLES["iid"])]
    fig, axes = base_figure("Exact Born average and Monte Carlo: fixed-half outcomes")
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5,.948), ncol=3, frameon=False)
    for row, L in enumerate(SIZES):
        for col, obs in enumerate(PLOT_OBSERVABLES):
            ax = axes[row, col]
            ax.plot(BETAS, [table[L,b,"exact",obs]["mean"] for b in BETAS], color="#242424", lw=1.6, zorder=1)
            for method in ("local", "iid"):
                values = [table[L,b,method,obs] for b in BETAS]
                ax.errorbar(BETAS, [v["mean"] for v in values], yerr=[2*v["SE"] for v in values],
                            linestyle="none", capsize=3, elinewidth=1, zorder=3, **STYLES[method])
            if col == 0:
                ax.set_ylabel("Mean Renyi-2 entropy (natural log)")
    fig.text(.5,.08,"30,000 samples per method and case; bars = 2 empirical block-SE. Exact lines connect the calculated beta values.",ha="center",fontsize=10)
    fig.text(.5,.055,"Overlapping means do not establish reliable uncertainty: the L2, beta=3 IID run misses rare high-entropy outcomes (see residual figure).",ha="center",fontsize=10)
    fig.text(.5,.03,"L2 uses test regions (A: 1x2, B: 1x1, C: 2x1); L3 uses A: 1x3, B: 1x2, C: 3x1. No scaling coefficient is fitted.",ha="center",fontsize=10)
    save(fig,"exact_born_means")

    indexed = {(r["L"],r["beta"],r["method"],r["observable"]):r for r in records}
    fig, axes = base_figure("Monte Carlo residuals relative to exact Born averages")
    residual_legend=[Line2D([],[],ls="none",label="Local / empirical block-SE",**STYLES["local"]),
        Line2D([],[],ls="none",label="IID / empirical block-SE",**STYLES["iid"]),
        Line2D([],[],ls="none",label="IID / exact-distribution SE",**STYLES["iid_oracle"])]
    fig.legend(handles=residual_legend, loc="upper center", bbox_to_anchor=(.5,.948), ncol=3, frameon=False)
    for row, L in enumerate(SIZES):
        for col, obs in enumerate(PLOT_OBSERVABLES):
            ax = axes[row,col]
            ax.axhline(0,color="#242424",lw=1)
            for reference in (-2,2):
                ax.axhline(reference,color="#888888",ls="--",lw=.8)
            largest = 2.5
            for method in ("local","iid"):
                selected = [indexed[L,b,method,obs] for b in BETAS]
                valid = [r for r in selected if r["standardized_residual"] is not None]
                ax.plot([r["beta"] for r in valid], [r["standardized_residual"] for r in valid],
                        linestyle="none", zorder=3, **STYLES[method])
                largest = max(largest, *(abs(r["standardized_residual"])+.4 for r in valid))
            oracle=[indexed[L,b,"iid",obs] for b in BETAS if b>0]
            ax.plot([r["beta"] for r in oracle],[r["oracle_iid_standardized_residual"] for r in oracle],
                    ls="none",zorder=2,**STYLES["iid_oracle"])
            largest=max(largest,*(abs(r["oracle_iid_standardized_residual"])+.4 for r in oracle))
            if L==2:
                ax.set_yscale("symlog",linthresh=3,linscale=1)
                ticks=[t for t in (-100,-10,-2,0,2,10,100) if abs(t)<=largest*1.15]
                ax.set_yticks(ticks,labels=[str(t) for t in ticks])
            margin=1.8 if L==2 else 1.15
            ax.set_ylim(-largest*margin,largest*margin)
            ax.text(0,.035,r"$\beta=0$: zero variance",transform=ax.get_xaxis_transform(),ha="left",fontsize=8.8,color="#555555")
            if col == 0:
                ax.set_ylabel(r"$(\overline{S}_{MC}-\overline{S}_{exact})/SE_{MC}$")
    fig.text(.5,.08,"L2 y-axis is symmetric-log (linear within |z| <= 3), preserving rare-tail outliers. L3 y-axis is linear; +/-2 lines are reference levels.",ha="center",fontsize=10)
    fig.text(.5,.055,"Exact-distribution SE = sqrt(Var_exact / 30,000) for IID only; a finite sample can miss rare outcomes and underestimate its empirical SE.",ha="center",fontsize=10)
    fig.text(.5,.03,"At beta=0 variance is zero: z is undefined and omitted. These multiple residuals are diagnostics, not a simultaneous confidence test.",ha="center",fontsize=10)
    save(fig,"exact_born_residuals")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,default=DATA/"exact_born_manifest.toml")
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True,parents=True)
    FIG.mkdir(exist_ok=True)
    manifest=verify_manifest(args.manifest.resolve())
    table=read_summary()
    audit,moments=audit_raw(table)
    records=comparisons(table,moments)
    tail_audit=audit_tail_diagnostic(moments,records)
    write_csv(OUT/"comparison.csv",records)
    write_csv(OUT/"beta_table.csv",beta_table(table,moments))
    reproduction=compare_reproduction_snapshot(manifest)
    make_plots(table,records)
    tracked(Path(__file__))
    defined=[r for r in records if r["standardized_residual"] is not None]
    max_row=max(defined,key=lambda r:abs(r["standardized_residual"]))
    summary=dict(status="PASS",files=FILES,summary_rows=len(table),comparison_rows=len(records),
        ensemble="fixed_half",sizes=list(SIZES),betas=list(BETAS),
        max_abs_standardized_residual=abs(max_row["standardized_residual"]),maximum_residual_case=max_row,
        max_abs_oracle_iid_residual=max(abs(r["oracle_iid_standardized_residual"]) for r in records if r["oracle_iid_standardized_residual"] is not None),
        zero_variance_rows=sum(r["standardized_residual"] is None for r in records),independent_raw_audit=audit,
        independent_tail_audit=tail_audit,
        reproduction_validation=reproduction,
        seeds=[run["seed"] for run in manifest["runs"]],
        conventions=dict(exact="All half-sized E subsets, with normalized Born weights; no Monte Carlo uncertainty.",
            uncertainty="Fixed block length 100, blocks computed within each independent local chain or IID sequence; displayed bars are +/-2 block-SE.",
            residual="(MC mean - exact mean)/MC block-SE; undefined for beta=0 zero variance. Multiple residuals are diagnostic, not one predeclared hypothesis test.",
            oracle_iid_uncertainty="Exact population variance divided by IID sample count, square-rooted; uses the full distribution, independently recomputed from every exact trajectory. Not applicable as an SE for correlated local samples.",
            rare_tail="L2 beta3 IID has a large residual in empirical-SE units because rare high-entropy outcomes are unobserved; compare exact-distribution IID SE, and do not claim all results agree within empirical error bars.",
            entropy="Born average of each normalized trajectory's Renyi-2 entropy, not entropy of the average density matrix.",
            geometry="L2 test partitions A1x2/B1x1/C2x1; L3 A1x3/B1x2/C3x1.",
            differences="A_minus_B and D are computed per trajectory before their uncertainty; D=(2A+C)/3-B.",
            scope="Finite-size numerical verification; A-B and D are not extracted corner coefficients. full_local is a separate protocol and is never mixed here."))
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({k:summary[k] for k in ("status","summary_rows","comparison_rows","max_abs_standardized_residual","zero_variance_rows")},indent=2))


if __name__=="__main__":
    main()
