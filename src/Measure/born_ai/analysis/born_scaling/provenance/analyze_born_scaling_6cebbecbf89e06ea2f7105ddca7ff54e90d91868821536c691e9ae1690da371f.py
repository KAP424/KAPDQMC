"""Independent analysis for the original-local Born scan.

No Monte Carlo is launched. Inputs require complete, hash-bound manifests.
Only this study's new analysis and figure directories are written. Production
draws are selected by their frozen plan, never mixed with pilot observations.

Diagnostics: rank-normalized and folded split Rhat, split-chain bulk/tail ESS,
raw-observable ESS, independent-chain batch means, cumulative/doubling checks,
and explicitly specified tail counts. A finite diagnostic cannot certify that
all chains have observed a rare contribution.

Source for rank/folding/localized diagnostics:
Vehtari et al. (2021), https://arxiv.org/abs/1903.08008 .
ESS below deliberately caps g>=1 (ESS<=draw count) as a conservative convention;
this is not a claim that IID sampling is a universal variance lower bound.
"""
from pathlib import Path
from statistics import NormalDist
import argparse
import csv
import hashlib
import io
import json
import math
import os
import tomllib
from datetime import datetime, timezone

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np

ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / "analysis" / "born_scaling"
FIGURES = ROOT / "figures" / "born_scaling"
BURNINS = (32, 64, 128, 256, 512)
BLOCK_LENGTHS = (16, 32, 64, 128)
PRECISION_TARGETS = {"SA": .01, "D": .003}
ANALYSIS_PROTOCOL = {
    "burnin_candidates": BURNINS,
    "screen_rhat_max": 1.01, "screen_min_bulk_tail_ESS": 400,
    "block_lengths": BLOCK_LENGTHS, "minimum_blocks_per_chain": 8,
    "minimum_block_retained_fraction": .95,
    "MCSE_rule": "Maximum of raw split-chain ESS MCSE and eligible block-mean SEs; not a confidence bound",
    "burnin_rule": "Earliest candidate whose SA and D screens pass at it and all larger candidates; drift/block stability still require inspection",
    "allocation_rule": "ceil_power_of_two(retained_draws_per_chain * (MCSE/target)^2), min 2048, cap 32768; no theoretical-mean target",
    "targets": PRECISION_TARGETS,
    "production_burnin": 256,
    "large_fit_residual_reduced_chi_square": 3.0,
    "residual_flag_role": "Descriptive misfit marker fixed before whole-grid analysis; not a strict p-value or point-exclusion rule",
    "tail_thresholds": {"SA": [.01, .1], "absolute_D": [.01, .1]},
    "caveat": "Finite-chain diagnostics cannot exclude a rare contribution missed by all chains; selected burn-in is provisional for independent production",
}
PROPOSED_COLUMNS = ("L", "beta", "mu", "chain", "seed", "start", "sweep",
    "record_index", "measure_every", "SA", "SB", "SC", "D", "accepted", "proposed", "log_weight")
GROUP_KEYS = ("L", "beta", "mu")
OBSERVABLES = ("SA", "SB", "SC", "D")
FIGURE_STYLES = (
    dict(marker="o", mfc="white", ms=9, ls="-", color="#286B9C"),
    dict(marker="s", mfc="#D47728", ms=4.5, ls="--", color="#D47728"),
    dict(marker="^", mfc="white", ms=7, ls=":", color="#718535"),
    dict(marker="D", mfc="white", ms=5, ls="-.", color="#99618A"),
)


def as_chains(values):
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 4 or not np.isfinite(x).all():
        raise ValueError("Require at least two equal-length finite chains, each with >=4 draws")
    return x


def split_chains(values):
    x = as_chains(values)
    if x.shape[1] < 8:
        raise ValueError("Split diagnostics require >=8 draws in each original chain")
    half = x.shape[1] // 2
    # An odd middle draw is omitted only from split diagnostics, not from means.
    return np.concatenate((x[:, :half], x[:, -half:]), axis=0)


def rank_normalize(values):
    x = np.asarray(values, dtype=float)
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts)-counts
    ranks = starts+(counts+1)/2  # Average ranks for ties, counted from one.
    probabilities = (ranks-3/8)/(x.size+1/4)
    normal = NormalDist()
    z = np.array([normal.inv_cdf(float(p)) for p in probabilities])
    return z[inverse].reshape(x.shape)


def variance_components(chains):
    x = as_chains(chains)
    m, n = x.shape
    within = float(np.var(x, axis=1, ddof=1).mean())
    between = float(n*np.var(x.mean(axis=1), ddof=1))
    var_plus = (n-1)/n*within+between/n
    return within, between, var_plus


def classic_rhat(chains):
    within, between, var_plus = variance_components(chains)
    if within == 0:
        return None if between == 0 else math.inf
    return math.sqrt(var_plus/within)


def fft_autocovariance(chains):
    x = as_chains(chains)
    centered = x-x.mean(axis=1, keepdims=True)
    n = x.shape[1]
    fft_size = 1 << (2*n-1).bit_length()
    spectrum = np.fft.rfft(centered, n=fft_size, axis=1)
    return np.fft.irfft(spectrum*spectrum.conj(), n=fft_size, axis=1)[:, :n]/n


def multichain_ess(chains):
    """Between/within-chain autocorrelation estimate with Geyer monotone pairs.

    Uses all available lags, not a hard-coded lag truncation. The final variance
    factor is conservatively capped at one, explicitly allowing no ESS>N claim.
    """
    x = as_chains(chains)
    m, n = x.shape
    within, _, var_plus = variance_components(x)
    if var_plus == 0:
        return dict(ESS=None, variance_factor=None, variance_plus=0.0, status="observed_constant")
    autocov = fft_autocovariance(x).mean(axis=0)
    rho = 1-(within-autocov)/var_plus
    rho[0] = 1.0
    pairs = []
    previous = math.inf
    for lag in range(0, n-1, 2):
        pair = float(rho[lag]+rho[lag+1])
        if pair <= 0:
            break
        pair = min(pair, previous)
        pairs.append(pair)
        previous = pair
    factor = max(1.0, -1+2*sum(pairs))
    return dict(ESS=m*n/factor, variance_factor=factor, variance_plus=var_plus,
                paired_lags_used=2*len(pairs), status="estimated")


def diagnose_chains(values, *, deterministic=False, constant_tolerance=1e-12):
    x = as_chains(values)
    spread = float(np.ptp(x))
    if deterministic:
        if spread > constant_tolerance:
            raise ValueError("The declared deterministic observable varies beyond numerical tolerance")
        return dict(status="deterministic", mean=float(x.mean()), n_chains=x.shape[0],
            draws_per_chain=x.shape[1], observed_range=spread, rhat_rank=None, rhat_folded=None,
            rhat_max=None, ESS_bulk=None, ESS_tail=None, ESS_mean=None, MCSE_ess=0.0,
            note="No stochastic Rhat/ESS is defined; validate against the direct deterministic baseline")
    if spread <= constant_tolerance:
        return dict(status="observed_constant_unresolved", mean=float(x.mean()),
            n_chains=x.shape[0],draws_per_chain=x.shape[1],observed_range=spread,
            rhat_rank=None,rhat_folded=None,rhat_max=None,ESS_bulk=None,ESS_tail=None,
            ESS_mean=None,MCSE_ess=None,note="All chains may have missed the same rare tail")
    split = split_chains(x)
    z = rank_normalize(split)
    folded = rank_normalize(np.abs(split-np.median(split)))
    rank_rhat = classic_rhat(z)
    folded_rhat = classic_rhat(folded)
    bulk = multichain_ess(z)
    raw = multichain_ess(split)
    q05, q95 = np.quantile(split, (.05, .95))
    low = multichain_ess((split <= q05).astype(float))
    high = multichain_ess((split <= q95).astype(float))
    tail_values = [v["ESS"] for v in (low, high)]
    tail = min(tail_values) if all(v is not None for v in tail_values) else None
    rhats = [v for v in (rank_rhat,folded_rhat) if v is not None]
    return dict(status="diagnostics_available_not_convergence_proof", mean=float(x.mean()),
        n_chains=x.shape[0],draws_per_chain=x.shape[1],observed_range=spread,
        rhat_rank=rank_rhat,rhat_folded=folded_rhat,rhat_max=max(rhats) if rhats else None,
        ESS_bulk=bulk["ESS"],ESS_tail=tail,ESS_tail_lower=low["ESS"],ESS_tail_upper=high["ESS"],
        ESS_mean=raw["ESS"],MCSE_ess=math.sqrt(raw["variance_plus"]/raw["ESS"]) if raw["ESS"] else None,
        tail_quantile_05=float(q05),tail_quantile_95=float(q95),
        tail_status="estimated" if tail is not None else "quantile_indicator_constant_unresolved")


def batch_mean_error(values, block_length):
    """Blocks never cross chain boundaries. No incomplete block is hidden."""
    x = as_chains(values)
    length = int(block_length)
    if length < 1:
        raise ValueError("Positive block length required")
    complete = x.shape[1]//length
    if complete < 2:
        return dict(block_length=length,status="insufficient_complete_blocks",SE=None)
    retained = complete*length
    blocks = x[:, :retained].reshape(x.shape[0], complete, length).mean(axis=2)
    se = float(blocks.std(ddof=1)/math.sqrt(blocks.size))
    return dict(block_length=length,blocks_per_chain=complete,total_blocks=blocks.size,
        discarded_end_draws_per_chain=x.shape[1]-retained,block_mean=float(blocks.mean()),
        full_mean=float(x.mean()),SE=se,status="estimated_requires_block_length_stability")


def tail_counts(values, thresholds):
    x = as_chains(values)
    result = []
    for definition in thresholds:
        threshold = float(definition["threshold"])
        transform = definition["transform"]
        if transform not in ("identity","absolute"):
            raise ValueError("Explicit identity or absolute tail transformation required")
        hits = (np.abs(x) if transform=="absolute" else x) > threshold
        for chain in range(x.shape[0]):
            selected = np.flatnonzero(hits[chain])
            result.append(dict(chain_index=chain,threshold=threshold,transform=transform,
                n=x.shape[1],hits=int(hits[chain].sum()),
                first_hit_draw=int(selected[0]+1) if len(selected) else None,
                maximum_observed=float(x[chain].max()),minimum_observed=float(x[chain].min()),
                origin=definition["origin"],
                note="Zero observed hits is not a zero-probability bound for correlated chains"))
    return result


def cumulative_diagnostics(values, checkpoints, block_lengths, *, deterministic=False):
    x = as_chains(values)
    rows = []
    for checkpoint in checkpoints:
        n = int(checkpoint)
        if n > x.shape[1]:
            rows.append(dict(draws_per_chain=n,status="not_yet_available"))
            continue
        diag = diagnose_chains(x[:,:n],deterministic=deterministic)
        diag["blocks"]=[batch_mean_error(x[:,:n],b) for b in block_lengths]
        if n%2==0 and n//2>=4:
            first,second=x[:,:n//2],x[:,n//2:n]
            # A descriptive nonoverlapping-half drift, not a significance test:
            # neighboring halves can still be correlated and require adequate blocks.
            diag["first_half_mean"]=float(first.mean())
            diag["second_half_mean"]=float(second.mean())
            diag["half_mean_difference"]=float(second.mean()-first.mean())
            diag["second_half_blocks"]=[batch_mean_error(second,b) for b in block_lengths]
        rows.append(diag)
    return rows


MODEL_TERMS = {
    "D_constant": ("intercept",),
    "D_log": ("intercept","logL"),
    "D_log_invL": ("intercept","logL","invL"),
    "D_linear_log_diagnostic": ("intercept","L","logL"),
    "SA_area": ("intercept","L"),
    "SA_area_log": ("intercept","L","logL"),
    "SA_LlogL_diagnostic": ("intercept","L","LlogL"),
}


def finite_size_fit(L, means, standard_errors, model, *, L_min, min_residual_dof=3):
    """Fit one explicitly chosen model/window; never auto-select a log law.

    Requires all input statistical eligibility decisions to be made externally.
    The beta=0 deterministic baseline uses unweighted numerical least squares;
    its residual-based uncertainty is not a Monte Carlo error bar.
    """
    sizes=np.asarray(L,dtype=float)
    values=np.asarray(means,dtype=float)
    errors=np.asarray(standard_errors,dtype=float)
    keep=sizes>=L_min
    sizes,values,errors=sizes[keep],values[keep],errors[keep]
    if not np.isfinite(sizes).all() or np.any(sizes<=0) or not np.isfinite(values).all():
        raise ValueError("Invalid finite-size observations")
    if len(set(sizes))!=len(sizes):
        raise ValueError("Duplicate L in a fixed beta/mu fit")
    terms=MODEL_TERMS[model]
    if len(sizes)<len(terms)+min_residual_dof:
        return dict(model=model,L_min=L_min,status="insufficient_sizes",n_sizes=len(sizes),n_parameters=len(terms))
    if np.all(errors==0):
        weights=np.ones_like(errors)
        error_role="deterministic_unweighted_residuals_not_MC_uncertainty"
    elif np.isfinite(errors).all() and np.all(errors>0):
        weights=1/errors
        error_role="estimated_MCSE_weights_not_model_systematic_uncertainty"
    else:
        raise ValueError("Do not mix deterministic/unknown/zero MCSE with positive MCSE in one weighted fit")
    columns={"intercept":np.ones_like(sizes),"L":sizes,"logL":np.log(sizes),
             "invL":1/sizes,"LlogL":sizes*np.log(sizes)}
    design=np.column_stack([columns[t] for t in terms])
    weighted=design*weights[:,None]
    scales=np.linalg.norm(weighted,axis=0)
    normalized=weighted/scales
    normalized_coefficient,_,rank,_=np.linalg.lstsq(normalized,values*weights,rcond=None)
    if rank!=len(terms):
        return dict(model=model,L_min=L_min,status="rank_deficient")
    coefficient=normalized_coefficient/scales
    residual=values-design@coefficient
    # Known/estimated MCSE weights are not rescaled by the fitted residuals.
    # Model misspecification is reported separately as chi-square and cut drift.
    covariance = None if np.all(errors==0) else (np.linalg.inv(normalized.T@normalized)/scales[:,None]/scales[None,:])
    return dict(model=model,L_min=float(sizes.min()),L_max=float(sizes.max()),n_sizes=len(sizes),
        n_parameters=len(terms),residual_dof=len(sizes)-len(terms),status="finite_window_diagnostic_only",
        coefficient=dict(zip(terms,map(float,coefficient))),sizes=sizes.tolist(),residuals=residual.tolist(),
        weighted_residual_sum_squares=float(np.sum((residual*weights)**2)),
        reduced_chi_square=None if covariance is None else float(np.sum((residual*weights)**2)/(len(sizes)-len(terms))),
        coefficient_MCSE=None if covariance is None else dict(zip(terms,map(float,np.sqrt(np.diag(covariance))))),
        coefficient_covariance=None if covariance is None else covariance.tolist(),
        normalized_design_condition=float(np.linalg.cond(normalized)),error_role=error_role,
        interpretation="A good finite-window fit does not establish asymptotic logarithmic scaling")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    return Path(os.path.relpath(Path(path).resolve(), ROOT)).as_posix()


def bind_analysis_snapshot(dependencies):
    """Preserve the exact analysis source when later analysis modes are added."""
    source=Path(__file__);digest=sha256(source)
    directory=ANALYSIS/"provenance";directory.mkdir(parents=True,exist_ok=True)
    snapshot=directory/f"analyze_born_scaling_{digest}.py"
    if snapshot.exists():
        if sha256(snapshot)!=digest:raise ValueError("Analysis snapshot content changed")
    else:
        snapshot.write_bytes(source.read_bytes())
    dependencies[relative(snapshot)]=digest
    return dict(original_entrypoint=relative(source),executed_source_sha256=digest,source_snapshot=relative(snapshot))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        raise ValueError("Refuse to emit a headerless analysis table")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_design_csv(path,rows):
    """A frozen allocation can be verified again, never silently changed."""
    buffer=io.StringIO(newline="")
    writer=csv.DictWriter(buffer,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    content=buffer.getvalue().encode("utf-8")
    if Path(path).exists() and Path(path).read_bytes()==content:return
    frozen=ROOT/"data"/"born_scaling"/"production"/"production_plan.toml"
    if frozen.exists():
        plan=tomllib.loads(frozen.read_text(encoding="utf-8"))
        if relative(path)==plan["allocation_path"]:
            raise ValueError("Refuse to alter the allocation already bound by the production plan")
    Path(path).write_bytes(content)


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def verify_file_map(mapping, base=ROOT):
    verified = {}
    for name, expected in mapping.items():
        path = Path(name) if Path(name).is_absolute() else base / name
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Source hash mismatch: {path}")
        verified[relative(path)] = actual
    return verified


def plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": .18,
        "savefig.dpi": 180, "pdf.fonttype": 42})
    return plt


def save_figure(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "pdf"):
        path = FIGURES / f"{name}.{suffix}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    return paths


def state_baseline():
    """A separate deterministic audit, with no MC observations or mixed states."""
    receipt_path = ROOT / "analysis" / "born_scaling_state_audit.toml"
    receipt = tomllib.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["status"] != "PASS":
        raise ValueError("State audit has not passed")
    dependencies = verify_file_map(receipt["files"], receipt_path.parent)
    dependencies.update(verify_file_map(receipt["source_hashes"]))
    dependencies[relative(receipt_path)] = sha256(receipt_path)
    rows = read_csv(receipt_path.with_suffix(".csv"))
    primary = sorted((r for r in rows if r["state"] == "staggered_mu1e-8"), key=lambda r: int(r["L"]))
    legacy = sorted((r for r in rows if r["state"] == "legacy_syevd"), key=lambda r: int(r["L"]))
    smallmass = sorted((r for r in rows if r["state"] == "staggered_mu1e-9"), key=lambda r: int(r["L"]))
    if len(primary) != 8 or any(float(r["beta"]) != 0 for r in primary):
        raise ValueError("Unexpected deterministic baseline grid")
    original_primary = list(primary)
    large_receipt_path = ROOT / "data" / "born_scaling" / "beta0_mass_large_sizes.toml"
    large_receipt = tomllib.loads(large_receipt_path.read_text(encoding="utf-8"))
    if large_receipt["status"] != "PASS": raise ValueError("Extended beta-zero audit incomplete")
    dependencies.update(verify_file_map(large_receipt["files"]))
    dependencies.update(verify_file_map(large_receipt["source_hashes"]))
    dependencies[relative(large_receipt_path)] = sha256(large_receipt_path)
    for r in read_csv(large_receipt_path.with_suffix(".csv")):
        if float(r["mu"]) != 1e-8 or float(r["beta"]) != 0: raise ValueError("Mixed baseline family")
        primary.append(r)
    primary.sort(key=lambda r:int(r["L"]))
    sizes = np.array([int(r["L"]) for r in primary]); d = np.array([float(r["D"]) for r in primary])
    fits = []
    for upper in (24,36):
      for lower in (6, 9, 12,15,18):
        for model in ("D_log", "D_log_invL"):
            subset = sizes <= upper
            fit = finite_size_fit(sizes[subset], d[subset], np.zeros(sum(subset)), model, L_min=lower, min_residual_dof=1)
            if fit["status"] != "finite_window_diagnostic_only": continue
            # These short-window diagnostics are shown with their actual dof,
            # not promoted to the production fit-eligibility policy.
            fits.append(dict(state="staggered_mu1e-8", mu=1e-8, beta=0,
                model=model, L_min=lower, L_max=fit["L_max"], n_sizes=fit["n_sizes"],
                residual_dof=fit["residual_dof"], c_fit=fit["coefficient"]["logL"],
                intercept=fit["coefficient"]["intercept"], invL=fit["coefficient"].get("invL", 0),
                residual_rmse=float(np.sqrt(np.mean(np.square(fit["residuals"])))),
                role="deterministic finite-window diagnostic; no MC uncertainty or scaling certification"))
    plt = plotting()
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), constrained_layout=True)
    ax = axes[0, 0]
    keep = sizes >= 6
    ax.plot(np.log(sizes[keep]), d[keep], **FIGURE_STYLES[0], lw=0, label=r"Direct $\mu=10^{-8}$")
    grid = np.linspace(6, 36, 300)
    for model, color, linestyle, label in (("D_log", "#99618A", "--", r"$b+c\ln L$"),
            ("D_log_invL", "#D47728", "-", r"$b+c\ln L+d/L$")):
        fit = next(f for f in fits if f["model"] == model and f["L_min"] == 6 and f["L_max"]==36)
        prediction = fit["intercept"]+fit["c_fit"]*np.log(grid)+fit["invL"]/grid
        ax.plot(np.log(grid), prediction, color=color, ls=linestyle, label=label+f", c={fit['c_fit']:.4f}")
    ax.set(xlabel=r"$\ln L$", ylabel=r"$D=(2S_A+S_C)/3-S_B$", title="Chosen state: visible finite-size curvature")
    ax.legend(fontsize=9)
    ax = axes[0, 1]
    for model, style, label in (("D_log", FIGURE_STYLES[3], "log only"),
            ("D_log_invL", FIGURE_STYLES[1], "log + 1/L")):
        selected = [r for r in fits if r["model"] == model and r["L_max"]==36]
        ax.plot([r["L_min"] for r in selected], [r["c_fit"] for r in selected], **style, label=label)
        for r in selected:
            ax.annotate(f"dof {r['residual_dof']}", (r["L_min"], r["c_fit"]), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    ax.set(xlabel=r"Lower size cutoff $L_{\min}$ ($L_{\max}=36$)", ylabel=r"Finite-window $c_{\rm fit}$", title="Window/model spread is not a statistical CI", xticks=[6,9,12,15,18])
    ax.margins(y=.25); ax.legend(fontsize=9)
    ax = axes[1, 0]
    ax.plot(np.log(sizes), d, **FIGURE_STYLES[0], label=r"chosen $\mu=10^{-8}$")
    ax.plot(np.log([int(r["L"]) for r in legacy]), [float(r["D"]) for r in legacy], **FIGURE_STYLES[1], label="legacy LAPACK zero modes (L<=24)")
    ax.set(xlabel=r"$\ln L$", ylabel="D", title="Zero-mode comparison: separate state families")
    ax.legend(fontsize=9)
    ax = axes[1, 1]
    sa = np.array([float(r["S_A"]) for r in primary])
    ax.plot(sizes, sa, **FIGURE_STYLES[0], label=r"$S_A$, chosen state")
    ax.plot(sizes, [float(r["S_C"]) for r in primary], **FIGURE_STYLES[1], label=r"$S_C$, chosen state")
    ax.set(xlabel="L", ylabel="Second Renyi entropy", title="Strip entropy: linear L axis")
    ax.legend(fontsize=9)
    max_mass_d = float(max(abs(float(r["D"])-float(s["D"])) for r,s in zip(original_primary,smallmass)))
    fig.suptitle(r"Deterministic $\beta=0$ baseline | staggered $+\mu$ on A, $-\mu$ on B", fontsize=14, y=.995)
    fig.get_layout_engine().set(rect=(0, .065, 1, .86))
    fig.text(.5, .018, f"No Monte Carlo error bars. For L<=24, changing mu from 1e-8 to 1e-9 changes D by at most {max_mass_d:.2g}.\n"
        "Finite mu is gapped at asymptotically large L; these sizes satisfy mu L << 1. Fits are finite-window diagnostics.", ha="center", fontsize=9)
    paths = save_figure(fig, "beta0_state_diagnostics"); plt.close(fig)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    table = ANALYSIS / "beta0_state_fits.csv"; write_csv(table, fits)
    paths.append(table)
    analysis_source=bind_analysis_snapshot(dependencies)
    dependencies.update({relative(p): sha256(p) for p in paths})
    output = dict(status="PASS", role="deterministic baseline only", files=dependencies,analysis_source=analysis_source,
        primary_state="staggered_mu1e-8", mass_sensitivity_max_D=max_mass_d, fits=fits)
    write_json(ANALYSIS / "beta0_state_receipt.json", output)
    return {"status":"PASS", "figures":[relative(p) for p in paths if p.suffix in (".png", ".pdf")], "fits": fits}


def load_complete_stage(stage):
    directory = ROOT / "data" / "born_scaling" / stage
    manifest_path = directory / f"{stage}_manifest.toml"
    if not manifest_path.exists():
        raise ValueError(f"No complete {stage} manifest yet; partial chains are not final pilot evidence")
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("Incomplete input stage")
    dependencies = verify_file_map(manifest["files"])
    dependencies[relative(manifest_path)] = sha256(manifest_path)
    plan_path = directory / f"{stage}_plan.toml"
    plan = tomllib.loads(plan_path.read_text(encoding="utf-8"))
    if manifest["plan_sha256"] != sha256(plan_path) or plan["stage"]!=stage or manifest["stage"]!=stage:
        raise ValueError("Complete manifest is not bound to this exact stage plan")
    dependencies[relative(plan_path)] = sha256(plan_path)
    dependencies.update(verify_file_map(plan["source_sha256"]))
    if "allocation_path" in plan:
        dependencies.update(verify_file_map({plan["allocation_path"]:plan["allocation_sha256"]}))
        dependencies.update(verify_file_map(plan.get("pilot_evidence",{})))
    if "parent_plan" in plan:
        dependencies.update(verify_file_map({plan["parent_plan"]:plan["parent_plan_sha256"]}))
    if plan["state"] != "staggered" or plan["mu"] != 1e-8 or plan["burnin_discarded"] != 0:
        raise ValueError("Unexpected state family or pre-discarded records")
    for model in plan["models"]:
        dependencies.update(verify_file_map(model["files"]))
    groups = {}; all_seeds = []; max_d_error = 0.0
    for job in plan["jobs"]:
        run_id = job["run_id"]
        receipt_path = directory / f"{run_id}_receipt.toml"
        receipt = tomllib.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["status"] != "complete" or receipt["plan_sha256"] != sha256(plan_path):
            raise ValueError(f"Incomplete/unbound chain: {run_id}")
        if receipt["job"] != job or receipt["source_sha256"] != plan["source_sha256"]:
            raise ValueError(f"Chain source/parameter mismatch: {run_id}")
        dependencies[relative(receipt_path)] = sha256(receipt_path)
        dependencies.update(verify_file_map(receipt["files"]))
        if receipt["maximum_inverse_drift"] > plan["max_inverse_drift_tolerance"] or receipt["maximum_logweight_drift"] > plan["max_logweight_drift_tolerance"]:
            raise ValueError(f"Numerical drift threshold failed: {run_id}")
        path = directory / f"{run_id}_samples.csv"
        rows = read_csv(path)
        expected_count = job["sweeps"] // job["measure_every"]
        if len(rows) != expected_count or tuple(rows[0]) != tuple(plan["columns"]):
            raise ValueError(f"Raw schema/length mismatch: {run_id}")
        for index, row in enumerate(rows, 1):
            for field in ("L", "chain", "seed", "measure_every"):
                if int(row[field]) != job[field]: raise ValueError(f"Raw {field} mismatch")
            for field in ("beta", "mu"):
                if float(row[field]) != job[field]: raise ValueError(f"Raw {field} mismatch")
            for field in ("start", "stage", "run_id"):
                if row[field] != job[field]: raise ValueError(f"Raw {field} mismatch")
            if row["phase"] != plan["phase"] or int(row["record_index"]) != index or int(row["sweep"]) != index*job["measure_every"]:
                raise ValueError("Missing/reordered sweep or incompatible phase")
            proposed = int(row["proposed"]); accepted = int(row["accepted"])
            if proposed != 2*job["L"]**2*job["measure_every"] or not 0 <= accepted <= proposed:
                raise ValueError("Acceptance counts are not original swap attempts")
        x = np.array([[float(r[field]) for field in ("SA", "SB", "SC", "D", "log_weight")] for r in rows])
        if not np.isfinite(x).all(): raise ValueError("Nonfinite raw observation")
        error = float(np.max(np.abs((2*x[:,0]+x[:,2])/3-x[:,1]-x[:,3])))
        max_d_error = max(max_d_error, error)
        if error > 1e-11: raise ValueError("Per-sample covariance-preserving D identity failed")
        initial = read_csv(directory / f"{run_id}_initial.csv")
        if len(initial) != 1 or int(initial[0]["sweep"]) != 0:
            raise ValueError("Missing separate sweep-zero record")
        key = tuple(job[k] for k in GROUP_KEYS)
        groups.setdefault(key, []).append(dict(job=job, receipt=receipt, data=x,
            sweeps=np.array([int(r["sweep"]) for r in rows],dtype=np.int64),record_count=len(rows),
            accepted_total=sum(int(r["accepted"]) for r in rows),proposed_total=sum(int(r["proposed"]) for r in rows),initial=initial[0]))
        all_seeds.append(job["seed"])
    if len(all_seeds) != len(set(all_seeds)): raise ValueError("Repeated chain seed")
    if sum(c["record_count"] for chains in groups.values() for c in chains)!=manifest["total_records"]:
        raise ValueError("Manifest total differs from all planned records")
    for key, chains in groups.items():
        chains.sort(key=lambda item: item["job"]["chain"])
        if [c["job"]["start"] for c in chains] != ["random","uniform_A","uniform_B","02_stripe"]:
            raise ValueError("Missing/different four starts in one case")
        if len({c["job"]["model_manifest_sha256"] for c in chains}) != 1:
            raise ValueError("Mixed projectors in one case")
    return plan, groups, dependencies, max_d_error


def production_observations(stages):
    """No CSV globbing: exactly the cases and retained sweeps in frozen plans."""
    merged={};dependencies={};max_error=0.;plans=[];seeds=[];pilot_seeds=set()
    for stage in stages:
        if not stage.startswith("production"):
            raise ValueError("Production inference must not include any pilot/smallcheck stage")
        plan,groups,source_files,error=load_complete_stage(stage)
        if plan["analysis_burnin"]!=256:raise ValueError("Production burn-in differs from frozen protocol")
        if plan.get("update_kernel")!="fused_rank_updates":raise ValueError("Production kernel differs from validated frozen choice")
        plans.append(plan);dependencies.update(source_files);max_error=max(max_error,error)
        for pilot_manifest_name in plan.get("pilot_evidence",{}):
            pilot_manifest=tomllib.loads((ROOT/pilot_manifest_name).read_text(encoding="utf-8"))
            if "plan_path" not in pilot_manifest:
                # The production plan also binds the independent equivalence
                # check of the fused arithmetic kernel, not another pilot run.
                if pilot_manifest.get("status")!="PASS" or not pilot_manifest.get("original_sources_unchanged"):
                    raise ValueError("Unrecognized/unpassed production design evidence")
                dependencies.update(verify_file_map(pilot_manifest["source_sha256"]))
                dependencies.update(verify_file_map({pilot_manifest["pipeline_manifest"]:pilot_manifest["pipeline_manifest_sha256"]}))
                validation_run=tomllib.loads((ROOT/pilot_manifest["pipeline_manifest"]).read_text(encoding="utf-8"))
                dependencies.update(verify_file_map(validation_run["files"]))
                continue
            pilot_plan_path=ROOT/pilot_manifest["plan_path"]
            if sha256(pilot_plan_path)!=pilot_manifest["plan_sha256"]:raise ValueError("Pilot-evidence plan hash mismatch")
            dependencies[relative(pilot_plan_path)]=sha256(pilot_plan_path)
            pilot_plan=tomllib.loads(pilot_plan_path.read_text(encoding="utf-8"))
            pilot_seeds.update(j["seed"] for j in pilot_plan["jobs"])
        for key,chains in groups.items():
            if key in merged:raise ValueError("Repeated production case across plans")
            counts=set();grids=set()
            for chain in chains:
                job=chain["job"];burn=job["analysis_burnin"];keep=job["retained_sweeps"]
                if burn!=256 or job["sweeps"]!=burn+keep or keep<2048:
                    raise ValueError("Frozen production retention counts are inconsistent")
                selection=chain["sweeps"]>burn
                retained_sweeps=chain["sweeps"][selection]
                if len(retained_sweeps)!=keep or retained_sweeps[0]!=burn+1 or retained_sweeps[-1]!=burn+keep:
                    raise ValueError("Production retention must be exactly sweep > frozen burnin")
                chain["production_data"]=chain["data"][selection]
                counts.add(keep);grids.add(job["grid"]);seeds.append(job["seed"])
            if len(counts)!=1 or len(grids)!=1:raise ValueError("Incompatible chain design within case")
            merged[key]=chains
    if len(seeds)!=len(set(seeds)):raise ValueError("Repeated seed across production plans")
    if set(seeds)&pilot_seeds:raise ValueError("Production seed reused from pilot design evidence")
    return merged,dependencies,max_error,plans


def production_analysis(stages,plots=False):
    groups,dependencies,max_error,plans=production_observations(stages)
    output_dir=ANALYSIS/"production";output_dir.mkdir(parents=True,exist_ok=True)
    stats_rows=[];block_rows=[];covariance_rows=[];cumulative_rows=[];tail_rows=[];means=[];chain_rows=[]
    for (L,beta,mu),chains in sorted(groups.items()):
        key=dict(L=L,beta=beta,mu=mu,grid=chains[0]["job"]["grid"])
        raw=np.stack([c["production_data"] for c in chains])
        values={"SA":raw[:,:,0],"SB":raw[:,:,1],"SC":raw[:,:,2],"D":raw[:,:,3],"SA_minus_SC":raw[:,:,0]-raw[:,:,2]}
        mean=dict(**key,burnin=256,draws_per_chain=raw.shape[1],n_chains=4,deterministic=False)
        for observable,x in values.items():
            diag,blocks=error_summary(x)
            stats_rows.append(dict(**key,observable=observable,**diag))
            for b in blocks:block_rows.append(dict(**key,observable=observable,**b))
            mean[f"{observable}_mean"]=diag["mean"];mean[f"{observable}_MCSE"]=diag["MCSE_conservative"]
            mean[f"{observable}_screen_pass"]=diag["initial_screen_pass"]
            mean[f"{observable}_rhat_max"]=diag["rhat_max"]
            mean[f"{observable}_ESS_bulk"]=diag["ESS_bulk"];mean[f"{observable}_ESS_tail"]=diag["ESS_tail"]
            for ci,c in enumerate(chains):
                chain_rows.append(dict(**key,observable=observable,chain=c["job"]["chain"],seed=c["job"]["seed"],
                    start=c["job"]["start"],mean=float(x[ci].mean()),draws=len(x[ci]),burnin=256))
            if observable in PRECISION_TARGETS:
                mean[f"{observable}_absolute_precision_target_met"] = bool(diag["MCSE_conservative"] is not None and diag["MCSE_conservative"]<=PRECISION_TARGETS[observable])
                checkpoints=[n for n in (128,256,512,1024,2048,4096,8192,16384,32768) if n<=x.shape[1]]
                for n in checkpoints:
                    prefix=x[:,:n];d,_=error_summary(prefix)
                    cumulative_rows.append(dict(**key,observable=observable,**d,first_half_mean=float(prefix[:,:n//2].mean()),
                        second_half_mean=float(prefix[:,n//2:].mean()),half_mean_difference=float(prefix[:,n//2:].mean()-prefix[:,:n//2].mean())))
                for tail in tail_counts(x,[dict(threshold=t,transform="absolute" if observable=="D" else "identity",origin="fixed study threshold") for t in (.01,.1)]):
                    tail_rows.append(dict(**key,observable=observable,**tail))
        for b in BLOCK_LENGTHS:covariance_rows.append(dict(**key,**covariance_checks(raw[:,:,:3],b)))
        mean["sampling_screen_pass"]=mean["SA_screen_pass"] and mean["D_screen_pass"]
        mean["sampling_status"]="sampling_screen_pass_scaling_not_certified" if mean["sampling_screen_pass"] else "sampling_unresolved"
        means.append(mean)
    # Direct beta-zero values use only the primary family, and only the main
    # production sizes in comparative fits. Larger baseline sizes stay separate.
    baseline_path=ROOT/"analysis"/"born_scaling_state_audit.toml"
    baseline=tomllib.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline["status"]!="PASS":raise ValueError("Beta-zero audit has not passed")
    dependencies.update(verify_file_map(baseline["files"],baseline_path.parent))
    dependencies.update(verify_file_map(baseline["source_hashes"]))
    dependencies[relative(baseline_path)]=sha256(baseline_path)
    main_sizes=sorted({L for (L,beta,mu),chains in groups.items() if chains[0]["job"]["grid"]=="main"})
    for row in read_csv(baseline_path.with_suffix(".csv")):
        if row["state"]!="staggered_mu1e-8" or int(row["L"]) not in main_sizes:continue
        mean=dict(L=int(row["L"]),beta=0.,mu=1e-8,grid="main",burnin=0,draws_per_chain=0,n_chains=0,deterministic=True,
            sampling_screen_pass=True,sampling_status="deterministic_direct_baseline")
        for target,source in (("SA","S_A"),("SB","S_B"),("SC","S_C"),("D","D"),("SA_minus_SC","S_A_minus_S_C")):
            mean[f"{target}_mean"]=float(row[source]);mean[f"{target}_MCSE"]=0.;mean[f"{target}_screen_pass"]=True
            mean[f"{target}_rhat_max"]=None;mean[f"{target}_ESS_bulk"]=None;mean[f"{target}_ESS_tail"]=None
        means.append(mean)
    means.sort(key=lambda r:(r["beta"],r["L"]))
    fits=[];residuals=[];local_slopes=[]
    for beta in sorted({r["beta"] for r in means if r["grid"]=="main"}):
        selected=sorted((r for r in means if r["beta"]==beta and r["grid"]=="main"),key=lambda r:r["L"])
        if [r["L"] for r in selected]!=main_sizes:raise ValueError("Incomplete main L grid for one beta")
        for left,right in zip(selected[:-1],selected[1:]):
            delta=math.log(right["L"]/left["L"])
            local_slopes.append(dict(beta=beta,mu=1e-8,L_left=left["L"],L_right=right["L"],
                D_local_log_slope=(right["D_mean"]-left["D_mean"])/delta,
                MCSE=math.hypot(left["D_MCSE"],right["D_MCSE"])/delta if left["D_MCSE"] is not None and right["D_MCSE"] is not None else None,
                role="Adjacent-size finite-difference slope; errors use independent size runs, neighboring intervals share a point"))
        for observable,models in (("D",("D_log_invL","D_log","D_linear_log_diagnostic")),("SA",("SA_area","SA_area_log"))):
            for lower in (6,9):
                fit_input=[r for r in selected if r["L"]>=lower]
                for model in models:
                    result=finite_size_fit([r["L"] for r in selected],[r[f"{observable}_mean"] for r in selected],
                        [r[f"{observable}_MCSE"] for r in selected],model,L_min=lower)
                    base=dict(beta=beta,mu=1e-8,observable=observable,model=model,L_min=lower,L_max=max(main_sizes),
                        sampling_screen_pass=all(r["sampling_screen_pass"] for r in fit_input),
                        interpretation="Finite-window fit only; statistical parameter SE excludes model/window systematic error")
                    flat=dict(**base,fit_status=result["status"],n_sizes=len(fit_input),residual_dof=result.get("residual_dof"),
                        chi_square=result.get("weighted_residual_sum_squares") if beta>0 else None,
                        reduced_chi_square=result.get("reduced_chi_square"),normalized_design_condition=result.get("normalized_design_condition"),
                        residual_rmse=float(np.sqrt(np.mean(np.square(result["residuals"])))) if "residuals" in result else None)
                    flat["large_fit_residual"]=bool(result.get("reduced_chi_square") is not None and result["reduced_chi_square"]>3.)
                    for term,value in result.get("coefficient",{}).items():
                        flat[f"coefficient_{term}"]=value
                        flat[f"MCSE_{term}"]=None if result["coefficient_MCSE"] is None else result["coefficient_MCSE"][term]
                    fits.append(flat)
                    for r,residual in zip(fit_input,result.get("residuals",[])):
                        residuals.append(dict(**base,L=r["L"],observed=r[f"{observable}_mean"],MCSE=r[f"{observable}_MCSE"],
                            residual=residual,fitted=r[f"{observable}_mean"]-residual,
                            standardized_residual=residual/r[f"{observable}_MCSE"] if r[f"{observable}_MCSE"]>0 else None))
    table_map={"means.csv":means,"diagnostics.csv":stats_rows,"chain_means.csv":chain_rows,"block_errors.csv":block_rows,
        "region_covariance.csv":covariance_rows,"cumulative_diagnostics.csv":cumulative_rows,"tail_counts.csv":tail_rows,
        "fits.csv":fits,"fit_residuals.csv":residuals,"D_local_log_slopes.csv":local_slopes}
    paths=[]
    for name,rows in table_map.items():path=output_dir/name;write_csv(path,rows);paths.append(path)
    if plots:
        paths+=plot_production(means,fits,residuals,cumulative_rows,stats_rows)
        paths+=plot_D_local_log_slopes(local_slopes)
    dependencies.update({relative(path):sha256(path) for path in paths});analysis_source=bind_analysis_snapshot(dependencies)
    summary=dict(status="PASS",stage="production",source_stages=stages,analysis_protocol=ANALYSIS_PROTOCOL,
        stochastic_cases=len(groups),deterministic_cases=sum(r["deterministic"] for r in means),chains=4*len(groups),
        retained_observations=sum(4*cs[0]["job"]["retained_sweeps"] for cs in groups.values()),
        per_sample_D_identity_max_error=max_error,main_sizes=main_sizes,files=dependencies,analysis_source=analysis_source,
        all_sampling_screen_pass=all(r["sampling_screen_pass"] for r in means),
        large_fit_residuals=[{k:r[k] for k in ("beta","observable","model","L_min","L_max","residual_dof","reduced_chi_square")} for r in fits if r["large_fit_residual"]],
        primary_D_large_residuals=[{k:r[k] for k in ("beta","L_min","L_max","residual_dof","reduced_chi_square")} for r in fits if r["large_fit_residual"] and r["model"]=="D_log_invL" and r["L_min"]==6],
        precision_misses=[dict(L=r["L"],beta=r["beta"],observable=o,MCSE=r[f"{o}_MCSE"],target=PRECISION_TARGETS[o]) for r in means if not r["deterministic"] for o in PRECISION_TARGETS if not r[f"{o}_absolute_precision_target_met"]],
        inference="No fit is declared a universal/asymptotic corner coefficient. Positive-beta MCSE, finite-window model error, and fixed-mu mass crossover are distinct.")
    write_json(output_dir/"summary.json",summary)
    return {k:v for k,v in summary.items() if k not in ("files","analysis_protocol")}


def plot_production(means,fits,residuals,cumulative_rows,stats_rows):
    plt=plotting();paths=[]
    # Size-specific markers/line styles stay visible at coinciding beta-zero or
    # strong-measurement values. No horizontal offsets alter the physical axes.
    colors=("#286B9C","#D47728","#718535","#99618A","#398B86","#AD4C52","#6F6D62")
    markers=("o","s","^","D","v","P","X")
    fig,axes=plt.subplots(1,2,figsize=(12.5,4.5),constrained_layout=True)
    for i,L in enumerate(sorted({r["L"] for r in means})):
        selected=sorted((r for r in means if r["L"]==L),key=lambda r:r["beta"])
        for ax,observable in zip(axes,("D","SA")):
            ax.errorbar([r["beta"] for r in selected],[r[f"{observable}_mean"] for r in selected],
                yerr=[r[f"{observable}_MCSE"] for r in selected],color=colors[i%7],marker=markers[i%7],
                mfc="white" if i%2==0 else colors[i%7],ms=8 if i%2==0 else 4.5,ls=("-","--",":","-.")[i%4],capsize=2,label=f"L={L}")
            ax.set(xlabel=r"Measurement strength $\beta$",ylabel=f"Born mean {observable}")
            failed=[r for r in selected if not r["sampling_screen_pass"]]
            if failed:ax.plot([r["beta"] for r in failed],[r[f"{observable}_mean"] for r in failed],"x",color="black",ms=10)
    for ax in axes:ax.legend(ncol=2,fontsize=8)
    fig.suptitle("Original-swap Born averages | error bars: conservative estimated MCSE",fontsize=13)
    paths+=save_figure(fig,"production_means_beta");plt.close(fig)
    for beta in sorted({r["beta"] for r in means if r["grid"]=="main"}):
        fig,axes=plt.subplots(2,2,figsize=(12,7.6),constrained_layout=True)
        selected=sorted((r for r in means if r["beta"]==beta and r["grid"]=="main"),key=lambda r:r["L"])
        L=np.array([r["L"] for r in selected]);grid=np.linspace(L.min(),L.max(),200)
        for column,observable,models in ((0,"D",("D_log_invL","D_log")),(1,"SA",("SA_area","SA_area_log"))):
            x=np.log(L) if observable=="D" else L
            axes[0,column].errorbar(x,[r[f"{observable}_mean"] for r in selected],yerr=[r[f"{observable}_MCSE"] for r in selected],
                fmt="o",color="#286B9C",mfc="white",ms=8,capsize=3,label="direct baseline" if beta==0 else "original local MC")
            for i,model in enumerate(models):
                f=next(r for r in fits if r["beta"]==beta and r["model"]==model and r["L_min"]==6)
                if f["fit_status"]!="finite_window_diagnostic_only":continue
                cols={"intercept":np.ones_like(grid),"L":grid,"logL":np.log(grid),"invL":1/grid}
                prediction=sum(f[f"coefficient_{term}"]*cols[term] for term in MODEL_TERMS[model])
                style=FIGURE_STYLES[1+i]
                name={"D_log_invL":"log + 1/L","D_log":"log only","SA_area":"area + constant","SA_area_log":"area + log + constant"}[model]
                chi="" if beta==0 else f", chi2/dof={f['reduced_chi_square']:.2g}"
                axes[0,column].plot(np.log(grid) if observable=="D" else grid,prediction,color=style["color"],ls=style["ls"],label=name+chi)
                rr=[r for r in residuals if r["beta"]==beta and r["model"]==model and r["L_min"]==6]
                axes[1,column].errorbar([math.log(r["L"]) if observable=="D" else r["L"] for r in rr],[r["residual"] for r in rr],
                    yerr=[r["MCSE"] for r in rr],**style,capsize=2,label=name)
            axes[0,column].set(ylabel=f"Born mean {observable}",title="Finite-window models, L=6–24")
            axes[1,column].set(xlabel=r"$\ln L$" if observable=="D" else "L",ylabel="Data minus fit")
            axes[1,column].axhline(0,color="black",lw=.7)
            for ax in axes[:,column]:ax.legend(fontsize=8)
        sampling_note="SAMPLING UNRESOLVED; " if not all(r["sampling_screen_pass"] for r in selected) else ""
        fig.suptitle(f"beta={beta:g}, mu=1e-8 | {sampling_note}finite-window diagnostics only",fontsize=13,y=.995)
        fig.get_layout_engine().set(rect=(0,.04,1,.88))
        fig.text(.5,.01,"No beta-zero MCSE. Positive-beta error bars exclude model/size-window error; fits and Lmin=9 checks are retained in tables.",ha="center",fontsize=9)
        paths+=save_figure(fig,f"production_scaling_beta_{str(beta).replace('.','p')}");plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12.5,4.7),constrained_layout=True)
    residual_label_shown=False;sampling_label_shown=False
    for i,(model,lower,label) in enumerate((("D_log_invL",6,"log + 1/L, Lmin6"),("D_log_invL",9,"log + 1/L, Lmin9"),("D_log",6,"log only, Lmin6"))):
        selected=sorted((r for r in fits if r["model"]==model and r["L_min"]==lower and "coefficient_logL" in r),key=lambda r:r["beta"])
        axes[0].errorbar([r["beta"] for r in selected],[r["coefficient_logL"] for r in selected],
            yerr=[r["MCSE_logL"] or 0 for r in selected],**FIGURE_STYLES[i],capsize=3,label=label)
        failed=[r for r in selected if not r["sampling_screen_pass"]]
        if failed:
            axes[0].plot([r["beta"] for r in failed],[r["coefficient_logL"] for r in failed],marker="X",ls="none",mfc="white",color="#B22222",ms=11,
                label="sampling unresolved" if not sampling_label_shown else None)
            sampling_label_shown=True
        poor=[r for r in selected if r.get("large_fit_residual",False)]
        if poor:
            axes[0].plot([r["beta"] for r in poor],[r["coefficient_logL"] for r in poor],"x",color="black",ms=9,mew=1.5,
                label="large fit residual (chi2/dof > 3)" if not residual_label_shown else None)
            residual_label_shown=True
    axes[0].set(xlabel=r"$\beta$",ylabel=r"Finite-window $c_{\rm fit}$",title="Finite-window coefficients; crosses flag poor fits")
    for i,(model,lower,label) in enumerate((("D_log_invL",6,"log + 1/L, Lmin6"),("D_log_invL",9,"log + 1/L, Lmin9"),("D_log",6,"log only, Lmin6"))):
        selected=sorted((r for r in fits if r["model"]==model and r["L_min"]==lower and r["reduced_chi_square"] is not None),key=lambda r:r["beta"])
        axes[1].plot([r["beta"] for r in selected],[r["reduced_chi_square"] for r in selected],**FIGURE_STYLES[i],label=label)
    axes[1].axhline(1,color="black",lw=.7,ls=":")
    axes[1].set(xlabel=r"$\beta$",ylabel="Reduced chi-square",title="Model mismatch is not absorbed into error bars",yscale="log")
    for ax in axes:ax.legend(fontsize=8)
    fig.suptitle("D fit diagnostics | statistical bars exclude model/window error; no shared coefficient imposed",fontsize=12)
    paths+=save_figure(fig,"production_D_fit_diagnostics");plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),constrained_layout=True)
    for i,(model,lower,label) in enumerate((("SA_area",6,"area, Lmin6"),("SA_area_log",6,"area + log, Lmin6"),("SA_area_log",9,"area + log, Lmin9"))):
        selected=sorted((r for r in fits if r["model"]==model and r["L_min"]==lower and "coefficient_L" in r),key=lambda r:r["beta"])
        axes[0].errorbar([r["beta"] for r in selected],[r["coefficient_L"] for r in selected],yerr=[r["MCSE_L"] or 0 for r in selected],
            **FIGURE_STYLES[i],capsize=3,label=label)
        if model=="SA_area_log":
            axes[1].errorbar([r["beta"] for r in selected],[r["coefficient_logL"] for r in selected],yerr=[r["MCSE_logL"] or 0 for r in selected],
                **FIGURE_STYLES[i],capsize=3,label=label)
        stochastic=[r for r in selected if r["reduced_chi_square"] is not None]
        axes[2].plot([r["beta"] for r in stochastic],[r["reduced_chi_square"] for r in stochastic],**FIGURE_STYLES[i],label=label)
    axes[0].set(xlabel=r"$\beta$",ylabel=r"Coefficient of $L$",title="Strip area-law coefficient")
    axes[1].set(xlabel=r"$\beta$",ylabel=r"Coefficient of additive $\ln L$",title="Strip correction; not a corner coefficient")
    axes[2].set(xlabel=r"$\beta$",ylabel="Reduced chi-square",title="Finite-window goodness of fit",yscale="log")
    axes[2].axhline(1,color="black",ls=":",lw=.7)
    for ax in axes:ax.legend(fontsize=8)
    fig.suptitle(r"$S_A=aL+b$ and $S_A=aL+b+c\ln L$ | this is not an $L\ln L$ fit",fontsize=13)
    paths+=save_figure(fig,"production_SA_fit_diagnostics");plt.close(fig)
    paths+=plot_production_sampling(means,cumulative_rows,stats_rows)
    return paths


def plot_D_local_log_slopes(rows):
    """Raw adjacent-size slopes, with their shared-endpoint dependence explicit."""
    plt=plotting()
    betas=sorted({float(r["beta"]) for r in rows})
    columns=4;panels=len(betas)+1;lines=math.ceil(panels/columns)
    fig,axes=plt.subplots(lines,columns,figsize=(16,3.55*lines),squeeze=False,constrained_layout=True)
    for ax,beta in zip(axes.flat,betas):
        selected=sorted((r for r in rows if float(r["beta"])==beta),key=lambda r:r["L_left"])
        middle=[math.sqrt(r["L_left"]*r["L_right"]) for r in selected]
        slopes=[r["D_local_log_slope"] for r in selected]
        errors=[r["MCSE"] for r in selected]
        if any(error is None for error in errors):
            raise ValueError("A local-slope panel has unresolved MCSE; do not draw absent errors as zero")
        style=FIGURE_STYLES[1] if beta==0 else FIGURE_STYLES[0]
        ax.errorbar(middle,slopes,yerr=errors,**style,lw=1,capsize=3)
        lower=min(r["L_left"] for r in selected);upper=max(r["L_right"] for r in selected)
        label=f"beta={beta:g}"+(" | deterministic baseline" if beta==0 else "")
        ax.set(xlabel=r"Interval midpoint $\sqrt{L_1L_2}$",ylabel=r"$(D_2-D_1)/\ln(L_2/L_1)$",
            title=label)
        ax.text(.04,.04,f"Adjacent sizes within L={lower}–{upper}",transform=ax.transAxes,fontsize=8,
            bbox=dict(facecolor="white",edgecolor="none",alpha=.8))
        ax.margins(x=.08,y=.15)
    note=axes.flat[len(betas)];note.axis("off")
    note.text(.02,.95,"Reading the raw slopes",transform=note.transAxes,va="top",fontsize=12,fontweight="bold")
    note.text(.02,.81,"Look for a size interval consistent\nwith a flat slope, within its errors.\n\n"
        "Each panel has its own vertical scale.\n\n"
        "Adjacent intervals share an endpoint:\nthese slope estimates are correlated.\n\n"
        "Lines connect finite differences;\nno plateau or common coefficient is fitted.",
        transform=note.transAxes,va="top",fontsize=10,linespacing=1.35)
    for ax in list(axes.flat)[panels:]:ax.axis("off")
    fig.suptitle(r"Raw local logarithmic slopes of Born-averaged $D$ | primary $\mu=10^{-8}$ state",fontsize=14,y=.995)
    fig.get_layout_engine().set(rect=(0,.075,1,.85))
    fig.text(.5,.019,r"Error propagation uses independent size runs: $\mathrm{SE}_{\rm slope}=\sqrt{\mathrm{SE}_1^2+\mathrm{SE}_2^2}/\ln(L_2/L_1)$."
        "\nBeta=0 is deterministic and has no Monte Carlo error bar; its size-dependent slope exposes finite-size curvature.",ha="center",fontsize=9)
    paths=save_figure(fig,"production_D_local_log_slopes");plt.close(fig)
    return paths


def plot_production_sampling(means,cumulative_rows,stats_rows):
    plt=plotting();paths=[]
    stochastic=[r for r in means if not r["deterministic"]]
    sizes=sorted({r["L"] for r in stochastic});betas=sorted({r["beta"] for r in stochastic})
    final_half={(r["L"],r["beta"],r["observable"]):r for r in cumulative_rows}
    specifications=(
        ("Max rank/folded Rhat (SA,D)",lambda r:max(r["SA_rhat_max"] or 0,r["D_rhat_max"] or 0),"Rhat; screening threshold 1.01"),
        ("Min bulk/tail ESS (SA,D)",lambda r:min(r["SA_ESS_bulk"] or 0,r["SA_ESS_tail"] or 0,r["D_ESS_bulk"] or 0,r["D_ESS_tail"] or 0),"ESS; screening threshold 400"),
        ("Direction contrast",lambda r:r["SA_minus_SC_mean"]/r["SA_minus_SC_MCSE"] if r["SA_minus_SC_MCSE"] else float("nan"),"mean(SA-SC) / estimated MCSE"),
        ("D absolute precision",lambda r:r["D_MCSE"]/.003,"Estimated MCSE / target 0.003"),
        ("SA absolute precision",lambda r:r["SA_MCSE"]/.01,"Estimated MCSE / target 0.01"),
        ("D disjoint-half drift",lambda r:abs(final_half[(r["L"],r["beta"],"D")]["half_mean_difference"])/.003,"Absolute half-mean difference / 0.003; not a z score"),
    )
    fig,axes=plt.subplots(2,3,figsize=(15.4,7.9),constrained_layout=True)
    for ax,(title,function,label) in zip(axes.flat,specifications):
        values=np.full((len(sizes),len(betas)),np.nan)
        for row in stochastic:values[sizes.index(row["L"]),betas.index(row["beta"])]=function(row)
        cmap=plt.get_cmap("viridis").copy();cmap.set_bad("#e8e8e8")
        image=ax.imshow(values,origin="lower",aspect="auto",cmap=cmap,interpolation="nearest")
        ax.set(xticks=range(len(betas)),xticklabels=[f"{b:g}" for b in betas],yticks=range(len(sizes)),yticklabels=sizes,
            xlabel=r"$\beta$",ylabel="L",title=title)
        ax.tick_params(axis="x",labelsize=8,rotation=45)
        ax.grid(False);fig.colorbar(image,ax=ax,shrink=.85,label=label)
    fig.suptitle("Production sampling checks | gray = not in the frozen grid",fontsize=13,y=.995)
    fig.get_layout_engine().set(rect=(0,.055,1,.86))
    fig.text(.5,.012,"All prescribed retained samples are shown. Direction/half-drift panels are diagnostics; no forced symmetry or independent-half z test.\n"
        "Rhat/ESS and estimated MCSE cannot exclude a tail missed by all chains; no finite-size scaling claim is made by these panels.",ha="center",fontsize=9)
    paths+=save_figure(fig,"production_sampling_diagnostics");plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12.8,4.8),constrained_layout=True)
    selected_cases=[(L,b) for L in (18,24) for b in (.5,1.5,3) if any(r["L"]==L and r["beta"]==b for r in stochastic)]
    extra_styles=(dict(marker="v",mfc="white",ms=7,ls="--",color="#398B86"),dict(marker="P",mfc="#AD4C52",ms=5,ls=":",color="#AD4C52"))
    styles=FIGURE_STYLES+extra_styles
    for ci,(L,beta) in enumerate(selected_cases):
        for ax,observable in zip(axes,("D","SA")):
            selected=sorted((r for r in cumulative_rows if r["L"]==L and r["beta"]==beta and r["observable"]==observable),key=lambda r:r["draws_per_chain"])
            ax.plot([r["draws_per_chain"] for r in selected],[r["MCSE_conservative"]/PRECISION_TARGETS[observable] for r in selected],
                **styles[ci],label=f"L={L}, beta={beta:g}")
            ax.set(xscale="log",yscale="log",xlabel="Retained sweeps per chain",ylabel="Estimated MCSE / absolute target",title=observable)
    for ax in axes:ax.axhline(1,color="black",ls=":",lw=.8);ax.legend(fontsize=8,ncol=2)
    fig.suptitle("Production prefix diagnostics | fixed final sample counts, no target-based stopping",fontsize=13)
    paths+=save_figure(fig,"production_error_vs_sweeps");plt.close(fig)
    return paths


def error_summary(values):
    x = as_chains(values)
    diagnostic = diagnose_chains(x)
    blocks = []
    for length in BLOCK_LENGTHS:
        result = batch_mean_error(x, length)
        result["eligible"] = (result.get("blocks_per_chain",0) >= 8 and
            result.get("discarded_end_draws_per_chain",x.shape[1])/x.shape[1] <= .05)
        blocks.append(result)
    estimates = [diagnostic["MCSE_ess"]] if diagnostic["MCSE_ess"] is not None else []
    estimates += [b["SE"] for b in blocks if b["eligible"] and b["SE"] is not None]
    eligible_se = [b["SE"] for b in blocks if b["eligible"] and b["SE"] is not None]
    diagnostic["MCSE_conservative"] = max(estimates) if estimates and diagnostic["MCSE_ess"] is not None else None
    diagnostic["eligible_block_SE_max_min_ratio"] = max(eligible_se)/min(eligible_se) if eligible_se and min(eligible_se)>0 else None
    diagnostic["initial_screen_pass"] = bool(diagnostic["rhat_max"] is not None and
        diagnostic["rhat_max"] <= 1.01 and diagnostic["ESS_bulk"] is not None and diagnostic["ESS_bulk"] >= 400 and
        diagnostic["ESS_tail"] is not None and diagnostic["ESS_tail"] >= 400)
    return diagnostic, blocks


def covariance_checks(values, block_length):
    # values: chain x time x (A,B,C); preserve each sample's cross-region covariance.
    x = np.asarray(values, dtype=float)
    n_blocks = x.shape[1]//block_length
    blocks = x[:,:n_blocks*block_length,:].reshape(x.shape[0],n_blocks,block_length,3).mean(axis=2).reshape(-1,3)
    covariance = np.cov(blocks, rowvar=False, ddof=1)/len(blocks)
    weights = np.array([2/3,-1,1/3])
    direct = float(np.var(blocks@weights, ddof=1)/len(blocks))
    propagated = float(weights@covariance@weights)
    if abs(direct-propagated)>1e-12: raise ValueError("D covariance propagation identity failed")
    result = dict(block_length=block_length,n_blocks=len(blocks),D_variance_direct=direct,D_variance_covariance=propagated)
    for i,a in enumerate(("SA","SB","SC")):
        for j,b in enumerate(("SA","SB","SC")):
            result[f"cov_mean_{a}_{b}"] = float(covariance[i,j])
    return result


def allocate_from_pilot(n_retained, mcse, target):
    if mcse is None:
        return dict(required_raw=None,required_power2=None,provisional_per_chain=32768,status="MCSE_unresolved_cap_is_not_a_precision_guarantee")
    needed = max(2048, n_retained*(mcse/target)**2)
    rounded = 2**math.ceil(math.log2(needed))
    return dict(required_raw=needed,required_power2=rounded,provisional_per_chain=min(32768,rounded),
        status="exceeds_provisional_cap" if rounded>32768 else "pilot_variance_extrapolation_not_guarantee")


def propose_production_allocation():
    """Design estimates only: never starts a run or changes an existing plan."""
    dependencies={};pilot_rows=[]
    for stage in ("pilot","pilot_extra"):
        # Reconstruct from validated raw input, so a mutable analysis receipt is
        # not a hidden prerequisite of the frozen production design.
        _,groups,files,_=load_complete_stage(stage);dependencies.update(files)
        for (L,beta,mu),chains in groups.items():
            raw=np.stack([c["data"][256:] for c in chains]);n=raw.shape[1]
            row=dict(L=L,beta=beta,mu=mu,n_retained=n,
                seconds_per_sweep=sum(c["receipt"]["elapsed_seconds"] for c in chains)/(4*chains[0]["job"]["sweeps"]))
            for observable,index in (("SA",0),("D",3)):
                diag,_=error_summary(raw[:,:,index])
                row[f"{observable}_MCSE"]=diag["MCSE_conservative"]
                row[f"{observable}_variance_scale"]=n*diag["MCSE_conservative"]**2
            pilot_rows.append(row)
    betas=sorted({r["beta"] for r in pilot_rows})
    def size_variance(L,beta,observable):
        rows=sorted((r for r in pilot_rows if r["beta"]==beta),key=lambda r:r["L"])
        sizes=np.array([r["L"] for r in rows]);v=np.maximum.accumulate([r[f"{observable}_variance_scale"] for r in rows])
        if L<sizes[0]:return float(v[0]*L/sizes[0]),"linear L/Lmin extrapolation below smallest measured size"
        if L>sizes[-1]:return float(v[-1]*L/sizes[-1]),"linear L/18 extrapolation above largest measured size"
        return float(np.interp(L,sizes,v)),"linear interpolation of nondecreasing measured variance envelope"
    def beta_variance(L,beta,observable):
        if beta in betas:adjacent=[beta]
        else:
            below=[b for b in betas if b<beta];above=[b for b in betas if b>beta]
            adjacent=([max(below)] if below else [])+([min(above)] if above else [])
        values=[size_variance(L,b,observable) for b in adjacent]
        return max(v[0] for v in values),adjacent,[v[1] for v in values]
    benchmark_path=ROOT/"data"/"born_scaling"/"benchmark_fused_original_swap.toml"
    benchmark=tomllib.loads(benchmark_path.read_text(encoding="utf-8"))
    if benchmark["status"]!="PASS" or sha256(benchmark_path.with_suffix(".csv"))!=benchmark["csv_sha256"]:
        raise ValueError("Fused timing audit not passed/hash-bound")
    dependencies[relative(benchmark_path)]=sha256(benchmark_path)
    dependencies[relative(benchmark_path.with_suffix(".csv"))]=sha256(benchmark_path.with_suffix(".csv"))
    timing={int(r["L"]):r["fused_seconds_per_sweep"] for r in benchmark["summary"]}
    pilot18beta15=next(r["seconds_per_sweep"] for r in pilot_rows if r["L"]==18 and r["beta"]==1.5)
    concurrent_factor=pilot18beta15/timing[18]
    # The old parallel pilot includes scheduling/cache contention. Below L18,
    # use those measured times as conservative estimates; at L>=18 show both
    # single-worker and pilot-concurrency-adjusted predictions.
    small_sizes=(6,12,18)
    small_times=[max(r["seconds_per_sweep"] for r in pilot_rows if r["L"]==L) for L in small_sizes]
    def estimate_seconds(L):
        if L>=18:return timing[L],timing[L]*concurrent_factor
        value=float(np.interp(L,small_sizes,small_times));return value,value
    design=[(L,b,"main") for L in (6,9,12,15,18,21,24) for b in (.5,1,1.5,2,2.5,3)]
    design +=[(L,b,"dense") for L in (6,12,18) for b in (.25,.75,1.25,1.75,2.25,2.75)]
    output_dir=ANALYSIS;output_dir.mkdir(parents=True,exist_ok=True)
    evidence=[];variants=[];paths=[]
    for multiplier,suffix in ((1.,""),(1.25,"_margin25")):
        allocation=[];seconds_ideal=0.;seconds_concurrent=0.
        for L,beta,grid in design:
            requirements=[]
            for observable in ("D","SA"):
                variance,adjacent,rules=beta_variance(L,beta,observable)
                requirement=variance*multiplier/PRECISION_TARGETS[observable]**2
                requirements.append(requirement)
                evidence.append(dict(L=L,beta=beta,grid=grid,observable=observable,variance_budget_multiplier=multiplier,
                    variance_scale=variance,required_continuous_draws_per_chain=requirement,target_MCSE=PRECISION_TARGETS[observable],
                    adjacent_beta_anchors=";".join(map(str,adjacent)),size_rule=";".join(rules),
                    beta_rule="maximum of nearest measured beta anchors; below beta=.5 uses .5 only",
                    scope="allocation extrapolation only; no assertion of true unmeasured variance"))
            keep=min(32768,2**math.ceil(math.log2(max(2048,*requirements))))
            allocation.append(dict(L=L,beta=beta,retained_sweeps=keep,burnin=256,grid=grid))
            ideal,concurrent=estimate_seconds(L)
            seconds_ideal+=4*(keep+256)*ideal;seconds_concurrent+=4*(keep+256)*concurrent
        path=output_dir/f"allocation_proposal{suffix}.csv";write_design_csv(path,allocation);paths.append(path)
        variants.append(dict(variance_budget_multiplier=multiplier,allocation_path=relative(path),cases=len(allocation),chains=4*len(allocation),
            retained_total=sum(4*r["retained_sweeps"] for r in allocation),burnin_total=4*256*len(allocation),
            per_chain_histogram={str(n):sum(r["retained_sweeps"]==n for r in allocation) for n in sorted({r["retained_sweeps"] for r in allocation})},
            ideal_four_worker_minutes=seconds_ideal/4/60,parallel_pilot_adjusted_four_worker_minutes=seconds_concurrent/4/60,
            theoretical_sequential_minutes=seconds_ideal/60,
            allocation=allocation))
    table=output_dir/"allocation_design_evidence.csv";write_csv(table,evidence);paths.append(table)
    dependencies.update({relative(p):sha256(p) for p in paths});analysis_source=bind_analysis_snapshot(dependencies)
    result=dict(status="DESIGN_CALCULATION_ONLY",variants=variants,files=dependencies,analysis_source=analysis_source,
        production_plan_already_frozen=(ROOT/"data"/"born_scaling"/"production"/"production_plan.toml").exists(),
        rule="Per observable n*MCSE^2; nondecreasing L envelope then linear size interpolation; outside measured sizes proportional to L; beta interpolation takes maximum of neighboring anchors",
        below_smallest_beta="beta=.25 uses beta=.5 as an allocation estimate, not a proven variance bound",
        parallel_timing_factor_at_L18=concurrent_factor,
        timing_note="Planning range only; four workers, 1 BLAS thread each. Larger-size/cache contention and beta-dependent acceptance can change wall time. No new MC launched.")
    write_json(output_dir/"allocation_proposal.json",result)
    return {"status":result["status"],"variants":[{k:v for k,v in r.items() if k!="allocation"} for r in variants],"rule":result["rule"],"timing_note":result["timing_note"]}


def progress_snapshot(stage="production"):
    """One explicitly requested sanity snapshot; never produces fit coefficients."""
    if not stage.startswith("production"):raise ValueError("Progress snapshot is for production only")
    directory=ROOT/"data"/"born_scaling"/stage;plan_path=directory/f"{stage}_plan.toml"
    plan=tomllib.loads(plan_path.read_text(encoding="utf-8"))
    if plan["update_kernel"]!="fused_rank_updates" or plan["analysis_burnin"]!=256:raise ValueError("Unexpected frozen production protocol")
    dependencies=verify_file_map(plan["source_sha256"]);dependencies[relative(plan_path)]=sha256(plan_path)
    grouped={};completed=0;max_d=0.;minimum_entropy=math.inf;max_inverse=0.;max_logweight=0.
    for job in plan["jobs"]:
        receipt_path=directory/f"{job['run_id']}_receipt.toml"
        if not receipt_path.exists():continue
        receipt=tomllib.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["status"]!="complete":continue
        if receipt["job"]!=job or receipt["plan_sha256"]!=sha256(plan_path) or receipt["source_sha256"]!=plan["source_sha256"]:
            raise ValueError("Unbound completed progress chain")
        dependencies[relative(receipt_path)]=sha256(receipt_path);dependencies.update(verify_file_map(receipt["files"]))
        rows=read_csv(directory/f"{job['run_id']}_samples.csv")
        if len(rows)!=job["sweeps"] or [int(r["sweep"]) for r in rows]!=list(range(1,job["sweeps"]+1)):
            raise ValueError("Progress sample grid mismatch")
        if tuple(rows[0])!=tuple(plan["columns"]):raise ValueError("Progress schema mismatch")
        for row in rows:
            if row["run_id"]!=job["run_id"] or float(row["mu"])!=1e-8 or float(row["beta"])!=job["beta"]:
                raise ValueError("Progress parameter mismatch")
        x=np.array([[float(r[k]) for k in ("SA","SB","SC","D")] for r in rows])
        if not np.isfinite(x).all():raise ValueError("Nonfinite progress entropy")
        d_error=float(np.max(abs((2*x[:,0]+x[:,2])/3-x[:,1]-x[:,3])))
        max_d=max(max_d,d_error);minimum_entropy=min(minimum_entropy,float(x[:,:3].min()))
        max_inverse=max(max_inverse,receipt["maximum_inverse_drift"]);max_logweight=max(max_logweight,receipt["maximum_logweight_drift"])
        if d_error>1e-11 or minimum_entropy<-1e-10 or max_inverse>1e-5 or max_logweight>1e-5:
            raise ValueError("Progress physical/numerical identity check failed")
        retained=x[256:]
        if len(retained)!=job["retained_sweeps"]:raise ValueError("Wrong progress retained count")
        grouped.setdefault((job["L"],job["beta"]),[]).append((job,retained))
        completed+=1
    summaries=[]
    for (L,beta),items in sorted(grouped.items()):
        if len(items)!=4:continue
        items.sort(key=lambda t:t[0]["chain"])
        if [item[0]["chain"] for item in items]!=[1,2,3,4]:raise ValueError("Progress chain duplication")
        data=np.stack([item[1] for item in items]);row=dict(L=L,beta=beta,mu=1e-8,retained_per_chain=data.shape[1])
        for observable,values in (("SA",data[:,:,0]),("D",data[:,:,3]),("SA_minus_SC",data[:,:,0]-data[:,:,2])):
            diag,_=error_summary(values)
            row[f"{observable}_mean"]=diag["mean"];row[f"{observable}_MCSE"]=diag["MCSE_conservative"]
            row[f"{observable}_rhat_max"]=diag["rhat_max"]
            row[f"{observable}_ESS_bulk"]=diag["ESS_bulk"]
            row[f"{observable}_ESS_tail"]=diag["ESS_tail"]
            row[f"{observable}_initial_screen_pass"]=diag["initial_screen_pass"]
            if observable in PRECISION_TARGETS:
                row[f"{observable}_precision_target_met"]=bool(diag["MCSE_conservative"] is not None and diag["MCSE_conservative"]<=PRECISION_TARGETS[observable])
        summaries.append(row)
    stamp=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir=ANALYSIS/"progress";output_dir.mkdir(parents=True,exist_ok=True)
    table=output_dir/f"{stage}_{stamp}.csv"
    if summaries:write_csv(table,summaries);dependencies[relative(table)]=sha256(table)
    analysis_source=bind_analysis_snapshot(dependencies)
    result=dict(status="PARTIAL_PROGRESS_SANITY_ONLY",created_utc=stamp,stage=stage,planned_chains=len(plan["jobs"]),
        completed_chains=completed,complete_four_chain_cases=len(summaries),partial_cases=sum(len(v)<4 for v in grouped.values()),
        maximum_D_identity_error=max_d,minimum_individual_entropy=minimum_entropy if completed else None,
        max_inverse_drift=max_inverse,max_logweight_drift=max_logweight,case_summaries=summaries,
        scope="No scaling fits, no final inference, and no replacement of complete-manifest production analysis",files=dependencies,analysis_source=analysis_source)
    write_json(output_dir/f"{stage}_{stamp}.json",result)
    return {k:v for k,v in result.items() if k not in ("files","analysis_source")}


def plot_progress_snapshot(snapshot_name):
    """Plot one immutable snapshot, without scanning or fitting live production."""
    snapshot_path=Path(snapshot_name)
    if not snapshot_path.is_absolute():snapshot_path=ROOT/snapshot_path
    snapshot=json.loads(snapshot_path.read_text(encoding="utf-8"))
    if snapshot["status"]!="PARTIAL_PROGRESS_SANITY_ONLY":
        raise ValueError("This plotting mode requires an explicitly partial immutable snapshot")
    dependencies=verify_file_map(snapshot["files"])
    dependencies[relative(snapshot_path)]=sha256(snapshot_path)
    mc_rows=[r for r in snapshot["case_summaries"] if r["L"]<=18]
    if not mc_rows:raise ValueError("No complete L<=18 cases in the selected snapshot")
    sizes=sorted({r["L"] for r in mc_rows});points=[]
    for row in mc_rows:
        if row["mu"]!=1e-8 or row["beta"]<=0:raise ValueError("Unexpected snapshot state or beta")
        points.append(dict(L=row["L"],beta=row["beta"],mu=row["mu"],SA_mean=row["SA_mean"],D_mean=row["D_mean"],
            SA_MCSE=row["SA_MCSE"],D_MCSE=row["D_MCSE"],source_kind="original_local_MC_from_fixed_progress_snapshot",
            source=relative(snapshot_path),retained_per_chain=row["retained_per_chain"],chains=4))
    audit_path=ROOT/"analysis"/"born_scaling_state_audit.toml"
    audit=tomllib.loads(audit_path.read_text(encoding="utf-8"))
    if audit["status"]!="PASS":raise ValueError("Deterministic beta-zero state audit has not passed")
    dependencies.update(verify_file_map(audit["files"],audit_path.parent))
    dependencies.update(verify_file_map(audit["source_hashes"]))
    dependencies[relative(audit_path)]=sha256(audit_path)
    for row in read_csv(audit_path.with_suffix(".csv")):
        if row["state"]!="staggered_mu1e-8" or int(row["L"]) not in sizes:continue
        if float(row["beta"])!=0 or float(row["mu"])!=1e-8:raise ValueError("Wrong deterministic baseline family")
        points.append(dict(L=int(row["L"]),beta=0.,mu=1e-8,SA_mean=float(row["S_A"]),D_mean=float(row["D"]),
            SA_MCSE=0.,D_MCSE=0.,source_kind="deterministic_beta0_state_audit_no_MC",
            source=relative(audit_path.with_suffix(".csv")),retained_per_chain=0,chains=0))
    if sum(p["beta"]==0 for p in points)!=len(sizes):raise ValueError("Missing/duplicate beta-zero point")
    points.sort(key=lambda r:(r["L"],r["beta"]))
    plt=plotting();fig,axes=plt.subplots(1,2,figsize=(13.4,6.3),constrained_layout=True)
    styles=FIGURE_STYLES+(dict(marker="v",mfc="white",ms=7,ls="--",color="#398B86"),)
    for i,L in enumerate(sizes):
        selected=[r for r in points if r["L"]==L]
        for ax,observable in zip(axes,("D","SA")):
            ax.errorbar([r["beta"] for r in selected],[r[f"{observable}_mean"] for r in selected],
                yerr=[r[f"{observable}_MCSE"] for r in selected],**styles[i],capsize=2.5,lw=1.2,label=f"L={L}")
    for ax in axes:
        ax.axvspan(-.065,.065,color="#9c9c9c",alpha=.14,zorder=0)
        ax.set(xlabel=r"Measurement strength $\beta$",xlim=(-.07,3.07),xticks=np.arange(0,3.01,.5))
        ax.legend(fontsize=9,ncol=2,title="System size",title_fontsize=9)
    axes[0].set(ylabel=r"$\langle D\rangle_{\rm Born}$",title=r"$D=(2S_2(A)+S_2(C))/3-S_2(B)$")
    axes[1].set(ylabel=r"$\langle S_2(A)\rangle_{\rm Born}$",title="Strip entropy")
    fig.suptitle("PROGRESS | L=21,24 PENDING | NO SCALING FIT",fontsize=15,fontweight="bold",color="#963C34",y=.995)
    fig.text(.5,.905,f"Frozen snapshot {snapshot['created_utc']} UTC: {snapshot['completed_chains']}/{snapshot['planned_chains']} chains; "
        f"{len(mc_rows)} complete cases at L <= 18. Primary staggered state: mu/t = 1e-8.",ha="center",fontsize=10)
    fig.get_layout_engine().set(rect=(0,.20,1,.63))
    fig.text(.5,.055,"beta > 0: original local Metropolis; error bars = max(raw-ESS MCSE, eligible block-mean SEs).\n"
        "beta = 0 (gray band): direct deterministic entropies from the state audit, with no Monte Carlo error bar.\n"
        "Lines connect measured points only. This snapshot contains no L=21/24 results and no size-scaling or corner-coefficient inference.",
        ha="center",fontsize=9,linespacing=1.4)
    output_dir=FIGURES/"progress";output_dir.mkdir(parents=True,exist_ok=True)
    paths=[]
    for suffix in ("png","pdf"):
        path=output_dir/f"{snapshot_path.stem}_means_beta.{suffix}"
        fig.savefig(path,bbox_inches="tight");paths.append(path)
    plt.close(fig)
    table=snapshot_path.parent/f"{snapshot_path.stem}_plot_points.csv";write_csv(table,points);paths.append(table)
    dependencies.update({relative(p):sha256(p) for p in paths});analysis_source=bind_analysis_snapshot(dependencies)
    receipt=dict(status="PARTIAL_PROGRESS_PLOT_ONLY",source_snapshot=relative(snapshot_path),
        snapshot_created_utc=snapshot["created_utc"],sizes=sizes,positive_beta_cases=len(mc_rows),deterministic_beta0_points=len(sizes),
        total_plotted_points=len(points),fits_performed=0,live_production_scanned=False,files=dependencies,analysis_source=analysis_source,
        uncertainty="Positive beta: snapshot conservative MCSE; beta zero: deterministic direct calculation, no statistical error.",
        scope="Display of an immutable partial snapshot; final analysis still requires the whole production manifest.")
    receipt_path=snapshot_path.parent/f"{snapshot_path.stem}_plot_receipt.json";write_json(receipt_path,receipt)
    return {k:v for k,v in receipt.items() if k not in ("files","analysis_source")}|{"figures":[relative(p) for p in paths if p.suffix in (".png",".pdf")],"receipt":relative(receipt_path)}


def analyze_stage(stage, plots=False):
    plan, groups, dependencies, max_error = load_complete_stage(stage)
    output_dir = ANALYSIS / stage; output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = []; block_rows = []; tail_rows = []; covariance_rows = []; cumulative_rows = []
    allocations = []; case_summaries = []; plot_paths = []
    for key, chains in sorted(groups.items()):
        L,beta,mu = key
        base = dict(L=L,beta=beta,mu=mu,stage=stage)
        raw = np.stack([c["data"] for c in chains])
        observations = {"SA":raw[:,:,0],"SB":raw[:,:,1],"SC":raw[:,:,2],"D":raw[:,:,3],"SA_minus_SC":raw[:,:,0]-raw[:,:,2]}
        by_burn = {}
        for burn in BURNINS:
            by_burn[burn] = {}
            for observable, values in observations.items():
                retained = values[:,burn:]
                diag, blocks = error_summary(retained)
                row = dict(**base,burnin=burn,observable=observable,**diag)
                # Other observables are reported, but SA/D alone define the requested screen.
                diagnostics.append(row); by_burn[burn][observable] = diag
                for block in blocks: block_rows.append(dict(**base,burnin=burn,observable=observable,**block))
            for length in BLOCK_LENGTHS:
                covariance_rows.append(dict(**base,burnin=burn,**covariance_checks(raw[:,burn:,:3],length)))
        eligible = [burn for burn in BURNINS if all(by_burn[b][o]["initial_screen_pass"] for b in BURNINS if b>=burn for o in ("SA","D"))]
        selected = min(eligible) if eligible else None
        # Root froze 256 for independent production after inspecting the pilot
        # screen; the earliest screened candidate remains a separate field.
        descriptive_burn = 256
        status = "provisional_burnin_screen_pass_inspect_drift_and_tails" if selected is not None else "sampling_unresolved"
        row = dict(**base,selected_burnin=selected,descriptive_burnin=descriptive_burn,status=status)
        acceptance = sum(c["accepted_total"] for c in chains)/sum(c["proposed_total"] for c in chains)
        row["acceptance_all_recorded"] = acceptance
        row["elapsed_chain_seconds_sum"] = sum(c["receipt"]["elapsed_seconds"] for c in chains)
        row["seconds_per_sweep_per_chain"] = row["elapsed_chain_seconds_sum"]/(raw.shape[0]*raw.shape[1])
        for observable in ("SA","D","SA_minus_SC"):
            final_diag = by_burn[descriptive_burn][observable]
            row[f"{observable}_mean"] = final_diag["mean"]
            row[f"{observable}_MCSE"] = final_diag["MCSE_conservative"]
            row[f"{observable}_rhat_max"] = final_diag["rhat_max"]
            row[f"{observable}_ESS_bulk"] = final_diag["ESS_bulk"]
            row[f"{observable}_ESS_tail"] = final_diag["ESS_tail"]
            if observable in PRECISION_TARGETS:
                allocation = allocate_from_pilot(raw.shape[1]-descriptive_burn, final_diag["MCSE_conservative"],PRECISION_TARGETS[observable])
                allocations.append(dict(**base,burnin_for_estimate=descriptive_burn,burnin_screen_status=status,observable=observable,
                    target_MCSE=PRECISION_TARGETS[observable],pilot_MCSE=final_diag["MCSE_conservative"],**allocation))
                row[f"{observable}_allocated_per_chain"] = allocation["provisional_per_chain"]
                row[f"{observable}_uncapped_per_chain"] = allocation["required_power2"]
        row["provisional_production_draws_per_chain"] = max(row["SA_allocated_per_chain"],row["D_allocated_per_chain"])
        row["estimated_sequential_seconds"] = 4*(row["provisional_production_draws_per_chain"]+descriptive_burn)*row["seconds_per_sweep_per_chain"]
        case_summaries.append(row)
        for observable in ("SA","D"):
            values = observations[observable][:,descriptive_burn:]
            definitions = [dict(threshold=t,transform="absolute" if observable=="D" else "identity",origin="analysis protocol fixed absolute threshold; not a known rare-event probability") for t in (.01,.1)]
            for result in tail_counts(values,definitions):
                result["start"] = chains[result["chain_index"]]["job"]["start"]
                tail_rows.append(dict(**base,burnin=descriptive_burn,observable=observable,**result))
            checkpoints = sorted(set([n for n in (64,128,256,512,1024,2048,4096,8192,16384,32768) if n<=values.shape[1]]+[values.shape[1]]))
            for n in checkpoints:
                subset = values[:,:n]; diag, blocks = error_summary(subset)
                first = subset[:,:n//2]; second = subset[:,n//2:]
                cumulative_rows.append(dict(**base,burnin=descriptive_burn,observable=observable,draws_per_chain=n,
                    **{k:v for k,v in diag.items() if k!="draws_per_chain"},first_half_mean=float(first.mean()),
                    second_half_mean=float(second.mean()),half_mean_difference=float(second.mean()-first.mean()),
                    half_drift_role="descriptive disjoint adjacent halves; no independence z-test"))
        if plots and stage!="smallcheck": plot_paths += plot_case(base,chains,observations,row,cumulative_rows)
    tables = {"burnin_diagnostics.csv":diagnostics,"block_errors.csv":block_rows,"tail_counts.csv":tail_rows,
        "region_covariance.csv":covariance_rows,"cumulative_diagnostics.csv":cumulative_rows,
        "allocation.csv":allocations,"case_summary.csv":case_summaries}
    for name, rows in tables.items(): write_csv(output_dir/name,rows)
    paths = [output_dir/name for name in tables]+plot_paths
    smallcheck = None
    if stage == "smallcheck":
        smallcheck, extra_paths, extra_dependencies = smallcheck_comparison(groups, output_dir)
        paths += extra_paths; dependencies.update(extra_dependencies)
        if plots:paths+=plot_smallcheck_exact(output_dir/"exact_comparison.csv")
    analysis_source=bind_analysis_snapshot(dependencies)
    dependencies.update({relative(path):sha256(path) for path in paths})
    summary = dict(status="PASS",stage=stage,scope="Complete-input pilot diagnostics and provisional independent-production allocations; no production scaling claim",
        cases=len(groups),chains=len(plan["jobs"]),raw_rows=sum(c["record_count"] for chains in groups.values() for c in chains),
        per_sample_D_identity_max_error=max_error,analysis_protocol=ANALYSIS_PROTOCOL,
        covariance_note="D and SA-SC errors use per-sample contrasts; SA/SB/SC covariance is retained and independently reconstructed from blocks",
        deterministic_beta0_note="Beta=0 belongs to its direct state audit; no stochastic ESS/MCSE is inferred",
        case_summaries=case_summaries,smallcheck_comparison=smallcheck,files=dependencies,analysis_source=analysis_source)
    write_json(output_dir/"summary.json",summary)
    return dict(status="PASS",stage=stage,cases=summary["cases"],raw_rows=summary["raw_rows"],case_summaries=case_summaries)


def smallcheck_comparison(groups, output_dir):
    """Use one prespecified 256-sweep discard for comparison with mass-specific exact sums."""
    path = ROOT / "data" / "born_scaling" / "exact_L3_mu1e-8.toml"
    receipt = tomllib.loads(path.read_text(encoding="utf-8"))
    if receipt["status"] != "PASS" or receipt["mu"] != 1e-8: raise ValueError("Wrong small-system exact reference")
    dependencies = verify_file_map(receipt["files"])
    dependencies.update(verify_file_map(receipt["source_hashes"]))
    dependencies[relative(path)] = sha256(path)
    exact = {(float(r["beta"]),r["observable"]):r for r in read_csv(path.with_suffix(".csv"))}
    exact_tails = read_csv(path.parent / "exact_L3_mu1e-8_tails.csv")
    rows=[];tails=[];burn=256
    for (L,beta,mu),chains in sorted(groups.items()):
        if L!=3: raise ValueError("Smallcheck reference applies only to L=3")
        model = tomllib.loads((ROOT/chains[0]["job"]["model_manifest"]).read_text(encoding="utf-8"))
        q_names=[name for name in model["files"] if name.endswith("_Q.bin")]
        if len(q_names)!=1 or model["files"][q_names[0]]!=receipt["ground_orbital_sha256"]:
            raise ValueError("Exact/MC ground-orbital hashes differ")
        data=np.stack([c["data"] for c in chains])[:,burn:,:]
        for index,observable in enumerate(("S_A","S_B","S_C","D")):
            values=data[:,:,index];stats,_=error_summary(values);reference=exact[(beta,observable)]
            truth=float(reference["mean"]);population_variance=float(reference["variance"])
            empirical=stats["MCSE_conservative"];iid_reference=math.sqrt(population_variance/values.size)
            rows.append(dict(L=L,beta=beta,mu=mu,observable=observable,burnin=burn,n=values.size,
                MC_mean=stats["mean"],exact_mean=truth,difference=stats["mean"]-truth,MCSE=empirical,
                standardized_empirical_difference=(stats["mean"]-truth)/empirical if empirical else None,
                exact_population_IID_reference_SE=iid_reference,
                exact_IID_role="Reference variance scale only; these samples are local MC, not IID",
                rhat_max=stats["rhat_max"],ESS_bulk=stats["ESS_bulk"],ESS_tail=stats["ESS_tail"]))
            for tail in exact_tails:
                if float(tail["beta"])!=beta or tail["observable"]!=observable:continue
                if tail["ranking"] == "absolute_observable":
                    transformed=np.abs(values)
                elif tail["ranking"] == "observable":
                    transformed=values
                else:
                    raise ValueError(f"Unknown exact tail ranking: {tail['ranking']}")
                threshold=float(tail["threshold"]);probability=float(tail["closed_event_probability"])
                for ci,c in enumerate(chains):
                    hits=int(np.sum(transformed[ci]>=threshold-1e-12))
                    tails.append(dict(L=L,beta=beta,mu=mu,observable=observable,chain=c["job"]["chain"],start=c["job"]["start"],
                        burnin=burn,n=values.shape[1],ranking=tail["ranking"],threshold=threshold,
                        nominal_tail_probability=float(tail["tail_probability_requested"]),exact_closed_event_probability=probability,
                        expected_stationary_hits=probability*values.shape[1],observed_hits=hits,
                        event_definition="transformed observable >= exact weighted-tail threshold - 1e-12; includes ties",
                        caveat="Expected count is stationary; no independent-binomial confidence claim for local MC"))
    table=output_dir/"exact_comparison.csv";tail_table=output_dir/"exact_tail_hits.csv"
    write_csv(table,rows);write_csv(tail_table,tails)
    return dict(fixed_burnin=burn,rows=len(rows),max_absolute_empirical_standardized_difference=max(abs(r["standardized_empirical_difference"]) for r in rows),
                interpretation="Small-L validation only; neither mixing proof at larger L nor Gaussian tail certification"),[table,tail_table],dependencies


def plot_smallcheck_exact(path):
    rows=read_csv(path);plt=plotting()
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),constrained_layout=True)
    for oi,observable in enumerate(("S_A","D")):
        selected=sorted((r for r in rows if r["observable"]==observable),key=lambda r:float(r["beta"]))
        exact_style=FIGURE_STYLES[2*oi];local_style=FIGURE_STYLES[2*oi+1]
        x=[float(r["beta"]) for r in selected]
        axes[0].plot(x,[float(r["exact_mean"]) for r in selected],**exact_style,label=f"{observable}: exact enumeration")
        axes[0].errorbar(x,[float(r["MC_mean"]) for r in selected],yerr=[float(r["MCSE"]) for r in selected],
            **local_style,capsize=3,label=f"{observable}: original local MC")
    for i,observable in enumerate(("S_A","S_B","S_C","D")):
        selected=sorted((r for r in rows if r["observable"]==observable),key=lambda r:float(r["beta"]))
        axes[1].plot([float(r["beta"]) for r in selected],[float(r["standardized_empirical_difference"]) for r in selected],
            **FIGURE_STYLES[i],label=observable)
    axes[0].set(xlabel=r"$\beta$",ylabel="Positive Born mean entropy / contrast",yscale="log",title="Same mass-selected Slater; fixed half-filled outcomes")
    axes[1].set(xlabel=r"$\beta$",ylabel="(MC mean - exact mean) / estimated MCSE",title="Reference residual, not a Gaussian tail test")
    axes[1].axhline(0,color="black",lw=.7)
    for ax in axes:ax.legend(fontsize=8)
    fig.suptitle("L=3 small-system validation | four original-swap chains, 16,128 retained sweeps each",fontsize=13,y=.995)
    fig.get_layout_engine().set(rect=(0,.10,1,.79))
    fig.text(.5,.012,"Exact enumeration has no MC error. Error bars retain each sample's A/B/C covariance; D=(2SA+SC)/3-SB.\n"
        "Small absolute error at beta=3 is compatible with sizable relative error; this check does not establish large-L convergence.",ha="center",fontsize=9)
    paths=save_figure(fig,"smallcheck_exact_comparison");plt.close(fig)
    return paths


def plot_case(base, chains, observations, summary, cumulative_rows):
    plt = plotting()
    fig, axes = plt.subplots(2,3,figsize=(15,7.5),constrained_layout=True)
    burn = summary["descriptive_burnin"]
    for ri, observable in enumerate(("SA","D")):
        values = observations[observable]
        for ci,chain in enumerate(chains):
            style = FIGURE_STYLES[ci]; sweeps = np.arange(1,values.shape[1]+1)
            label={"random":"random E","uniform_A":"E = all A","uniform_B":"E = all B","02_stripe":"striped E"}[chain["job"]["start"]]
            trace_style = {k:style[k] for k in ("color","ls")}
            axes[ri,0].plot(sweeps,values[ci],**trace_style,lw=.55,alpha=.65,label=label)
            retained = values[ci,burn:]; n = np.arange(1,len(retained)+1)
            # All cumulative values are plotted; sparse staggered markers preserve overlapping chains.
            axes[ri,1].plot(n, np.cumsum(retained)/n, **style, lw=1,
                markevery=(ci*11,max(1,len(n)//12)),label=label)
        axes[ri,0].axvline(burn,color="black",ls=":",lw=1)
        axes[ri,0].set(xlabel="Sweep (including warmup)",ylabel=observable,title=f"{observable}: raw traces, no rejection thinning")
        axes[ri,1].set(xlabel=f"Recorded sweeps after {burn} discarded",ylabel=f"Cumulative mean {observable}",title="Each start retained separately")
        select = [r for r in cumulative_rows if all(r[k]==base[k] for k in GROUP_KEYS) and r["observable"]==observable]
        axes[ri,2].plot([r["draws_per_chain"] for r in select],[r["MCSE_conservative"] for r in select],**FIGURE_STYLES[0],label="max(ESS, eligible blocks)")
        axes[ri,2].plot([r["draws_per_chain"] for r in select],[r["MCSE_ess"] for r in select],**FIGURE_STYLES[1],label="raw ESS MCSE")
        axes[ri,2].axhline(PRECISION_TARGETS[observable],color="#718535",ls=":",label="production precision target")
        axes[ri,2].set(xscale="log",yscale="log",xlabel="Retained sweeps per chain",ylabel="Pooled MCSE",title="Error estimate vs sample count")
        for ax in axes[ri]: ax.legend(fontsize=7,loc="best")
    label = "screen passed; production discard = 256" if summary["selected_burnin"] is not None else "UNRESOLVED: fixed 256 discard only for display"
    fig.suptitle(f"{base['stage']} | L={base['L']}, beta={base['beta']:g}, mu=1e-8 | {label}",fontsize=13,y=.995)
    fig.get_layout_engine().set(rect=(0,.065,1,.86))
    fig.text(.5,.015,"Four independent original-swap chains; starts refer to suppressed set E. MCSE excludes unknown model/rare-tail bias.\n"
        "Cumulative prefixes overlap; adjacent halves are descriptive checks, not independent significance tests.",ha="center",fontsize=9)
    name=f"{base['stage']}_L{base['L']}_beta{str(base['beta']).replace('.','p')}_convergence"
    paths=save_figure(fig,name);plt.close(fig)
    return paths


def self_test():
    rng=np.random.default_rng(621038)
    independent=rng.normal(size=(4,4096))
    diag=diagnose_chains(independent)
    assert diag["rhat_max"]<1.02 and diag["ESS_bulk"]>4000
    shifted=independent.copy();shifted[0]+=1.5
    assert diagnose_chains(shifted)["rhat_max"]>1.05
    unequal=independent.copy();unequal[0]*=.1
    assert diagnose_chains(unequal)["rhat_folded"]>1.05
    assert diagnose_chains(np.ones((4,20)))["status"]=="observed_constant_unresolved"
    assert diagnose_chains(np.ones((4,20)),deterministic=True)["ESS_mean"] is None
    small=independent[:,:32]
    actual=fft_autocovariance(small)
    centered=small-small.mean(axis=1,keepdims=True)
    direct=np.array([[np.dot(row[:32-k],row[k:])/32 for k in range(32)] for row in centered])
    assert np.max(abs(actual-direct))<1e-13
    tied=rank_normalize(np.array([[1.,1.,2.,3.],[3.,4.,4.,5.]]))
    assert tied[0,0]==tied[0,1] and tied[0,3]==tied[1,0]
    sizes=np.arange(6,30,3,dtype=float)
    fit=finite_size_fit(sizes, .4+ .1558*np.log(sizes),np.zeros(len(sizes)),"D_log",L_min=6)
    assert abs(fit["coefficient"]["logL"]-.1558)<1e-12
    weighted=finite_size_fit(sizes,.4+.1558*np.log(sizes),np.full(len(sizes),.01),"D_log",L_min=6)
    assert weighted["coefficient_MCSE"]["logL"]>0 and weighted["reduced_chi_square"]<1e-20
    direct_covariance=np.linalg.inv(np.column_stack((np.ones(len(sizes)),np.log(sizes))).T@np.column_stack((np.ones(len(sizes)),np.log(sizes))))*.01**2
    assert np.max(abs(np.asarray(weighted["coefficient_covariance"])-direct_covariance))<1e-13
    assert allocate_from_pilot(1792,.004947,.003)["provisional_per_chain"]==8192
    return dict(status="PASS",scope="Synthetic statistical/linear-algebra checks only; no physical MC or data plots",iid_diagnostics=diag)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--state-baseline",action="store_true")
    parser.add_argument("--stage",choices=("pilot","pilot_extra","smallcheck"))
    parser.add_argument("--production-stages",nargs="+",help="Complete frozen production stage directories; pilot stages rejected")
    parser.add_argument("--propose-allocation",action="store_true")
    parser.add_argument("--progress",action="store_true")
    parser.add_argument("--plot-progress",help="Path, relative to born_ai or absolute, to one immutable progress JSON snapshot")
    parser.add_argument("--plots",action="store_true")
    args=parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(),indent=2,allow_nan=False))
    elif args.state_baseline:
        print(json.dumps(state_baseline(),indent=2,allow_nan=False))
    elif args.stage:
        print(json.dumps(analyze_stage(args.stage,plots=args.plots),indent=2,allow_nan=False))
    elif args.production_stages:
        print(json.dumps(production_analysis(args.production_stages,plots=args.plots),indent=2,allow_nan=False))
    elif args.propose_allocation:
        print(json.dumps(propose_production_allocation(),indent=2,allow_nan=False))
    elif args.progress:
        print(json.dumps(progress_snapshot(),indent=2,allow_nan=False))
    elif args.plot_progress:
        print(json.dumps(plot_progress_snapshot(args.plot_progress),indent=2,allow_nan=False))
    else:
        parser.error("Choose --self-test, --state-baseline, or --stage pilot/smallcheck; production is not yet frozen")


if __name__=="__main__":
    main()
