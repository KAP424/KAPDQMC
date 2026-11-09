"""Sampling-only diagnostics; finite runs cannot certify worst-case mixing."""
from pathlib import Path
from collections import defaultdict
import csv,hashlib,json,math,tomllib,os
from contextlib import nullcontext
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent;DATA=ROOT/"data";FIG=ROOT/"figures";FIG.mkdir(exist_ok=True)
STARTS=("uniform_A","02_stripe","random");OBS=("imbalance","n02_fraction","log_weight","site_memory")
COLORS=("#286B9C","#D47728","#7A4CA1")
LINE_STYLES=({"marker":"o","ms":9,"mfc":"white","mew":1.5,"ls":"-"},
    {"marker":"s","ms":7,"mfc":"white","mew":1.4,"ls":"--"},
    {"marker":"^","ms":4.5,"mew":1.2,"ls":":"})
PERIODIC_STYLES=({"marker":"o","ms":6,"mfc":"white","ls":"-"},
    {"marker":"^","ms":9,"mfc":"white","mew":1.4,"ls":"--"},
    {"marker":"s","ms":4,"ls":":"},
    {"marker":"D","ms":5,"mfc":"white","ls":"-."},
    {"marker":"v","ms":4,"ls":(0,(5,2,1,2))})
FILES={}
def tracked(path):
    data=path.read_bytes();FILES[path.relative_to(ROOT).as_posix()]=hashlib.sha256(data).hexdigest();return data
def read(name):return list(csv.DictReader(tracked(DATA/name).decode().splitlines()))
def write(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (DATA/name).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    tracked(DATA/name)
def save(fig,name):
    for ext in ("png","pdf"):
        p=FIG/f"{name}.{ext}";fig.savefig(p,dpi=170,bbox_inches="tight");tracked(p)
    plt.close(fig)
def acf_ess(values):
    x=np.asarray(values,float);n=len(x);y=x-x.mean();v=float(np.dot(y,y)/n)
    if v<1e-28:return np.ones(1),1.,float(n),0.
    acf=np.correlate(y,y,"full")[n-1:]/(n*v)
    pairs=[];last=math.inf
    # Standard Geyer pairs: (rho_0+rho_1), (rho_2+rho_3), ... .
    for lag in range(0,n-1,2):
        pair=float(acf[lag]+acf[lag+1])
        if pair<=0:break
        pair=min(pair,last);pairs.append(pair);last=pair
    # Clamp the variance factor to >=1: do not claim ESS > n from negative ACF.
    tau=max(1.,-1+2*sum(pairs));ess=n/tau
    return acf,tau,ess,float(np.std(x,ddof=1)/math.sqrt(ess))
def rhat(chains):
    half=len(chains[0])//2
    xs=np.array([part for x in chains for part in (x[:half],x[-half:])]);W=np.mean(np.var(xs,axis=1,ddof=1));B=half*np.var(np.mean(xs,axis=1),ddof=1)
    if W<1e-28:return 1. if B<1e-28 else None
    return float(math.sqrt(((half-1)/half*W+B/half)/W))
def target_mask(L,kind):
    target=[]
    for y in range(1,L+1):
        for x in range(1,L+1):
            a=2*(x-1+(y-1)*L);b=a+1
            if kind=="uniform_A":target.append(a)
            elif kind=="AB_checker":target.append(a if (x+y)%2==0 else b)
            elif kind=="02_stripe" and x%2:target.extend((a,b))
            elif kind=="02_checker" and (x+y)%2==0:target.extend((a,b))
            elif kind=="mixed":
                cell=(x-1)%2+2*((y-1)%2)
                if cell==0:target.extend((a,b))
                elif cell==1:target.append(a)
                elif cell==2:target.append(b)
    return target
def target_probabilities():
    rows=[]
    with nullcontext():
        for L in (2,6,12,18):
            N=2*L*L;a=L*L
            G=np.frombuffer(tracked(DATA/f"local_projector_L{L}.bin"),dtype=np.float64).reshape((N,N),order="F")
            for kind in ("uniform_A","AB_checker","02_stripe","02_checker","mixed"):
                S=target_mask(L,kind);E=sorted(set(range(N))-set(S));spectrum=np.linalg.eigvalsh(G[np.ix_(E,E)])
                assert min(spectrum)>-1e-9 and max(spectrum)<1+1e-9
                spectrum=np.clip(spectrum,0,1);d=int(np.sum(spectrum>=1-1e-9))
                for beta in (0.,.25,.5,1.,2.,3.):
                    terms=np.array([2*(math.lgamma(a+1)-math.lgamma(j+1)-math.lgamma(a-j+1))-4*beta*j for j in range(a+1)])
                    high=max(terms);logZ=float(high+math.log(sum(np.exp(terms-high))))
                    lw=float(np.sum(np.log1p(math.expm1(-4*beta)*spectrum)))
                    rows.append(dict(L=L,beta=beta,target_pattern=kind,pattern_means="occupied target S; suppressed E is complement",
                        target_bits="".join("1" if i in S else "0" for i in range(N)),d_eigenvalues_near_one=d,eigenvalue_tolerance=1e-9,
                        log_weight=lw,log_probability=lw-logZ,log10_probability=(lw-logZ)/math.log(10),
                        strong_measurement_statement="w(E) proportional to exp(-4 beta d) asymptotically; not a fitted claim at beta<=3"))
    write("periodic_target_probabilities.csv",rows)
    return rows
def main():
    manifest=tomllib.loads(tracked(DATA/"local_experiment_manifest.toml").decode());iidmeta=tomllib.loads(tracked(DATA/"local_iid_manifest.toml").decode())
    assert manifest["status"]==iidmeta["status"]=="complete"
    executed_source=ROOT/manifest.get("executed_script_snapshot","local_update_experiments.jl")
    assert hashlib.sha256(tracked(executed_source)).hexdigest()==manifest["code_sha256"]
    tracked(ROOT/"local_update_experiments.jl")  # Current reproducible entry point, distinct from executed snapshot.
    assert hashlib.sha256(tracked(ROOT/"local_iid_reference.jl")).hexdigest()==iidmeta["script_sha256"]
    assert hashlib.sha256(tracked(ROOT/"BornSampler.jl")).hexdigest()==iidmeta["sampler_sha256"]
    for filename,digest in manifest["files"].items():assert hashlib.sha256(tracked(DATA/filename)).hexdigest()==digest
    assert hashlib.sha256(tracked(DATA/"local_iid_reference.csv")).hexdigest()==iidmeta["output_sha256"]
    traces=read("local_trajectories.csv");iid=read("local_iid_reference.csv");exact=read("local_exact_summary.csv");specific=target_probabilities()
    groups=defaultdict(list);references=defaultdict(list)
    for r in traces:groups[int(r["L"]),float(r["beta"]),r["start"]].append(r)
    for r in iid:references[int(r["L"]),float(r["beta"])].append(r)
    assert len(groups)==45 and all(len(rows)==manifest["sweeps"]+1 for rows in groups.values())
    assert len(references)==15 and all(len(rows)==iidmeta["n_per_setting"] for rows in references.values())
    summaries=[];acfrows=[];pooled=[];classrows=[]
    for (L,beta,start),rows in groups.items():
        kept=[r for r in rows if r["post_burnin"].lower()=="true"]
        ref=references[L,beta]
        for observable in OBS:
            x=np.array([float(r[observable]) for r in kept]);acf,tau,ess,se=acf_ess(x)
            refkey="memory_"+start if observable=="site_memory" else observable
            target=np.array([float(r[refkey]) for r in ref]);refmean=float(target.mean());refse=float(target.std(ddof=1)/math.sqrt(len(target)))
            combined=math.hypot(se,refse)
            summaries.append(dict(L=L,beta=beta,start=start,observable=observable,sweeps=manifest["sweeps"],burnin=manifest["burnin"],n=len(x),
                mean=float(x.mean()),sd=float(x.std(ddof=1)),tau_int_sweeps=tau,ESS=ess,MCSE=se,
                iid_n=len(target),iid_mean=refmean,iid_SE=refse,difference=float(x.mean())-refmean,
                difference_over_combined_SE=(float(x.mean())-refmean)/combined if combined else 0.,
                mean_acceptance=float(np.mean([float(r["acceptance"]) for r in kept])),
                max_inverse_drift=max(float(r["inverse_drift_before_refresh"]) for r in rows),
                max_logweight_drift=max(float(r["logweight_drift_before_refresh"]) for r in rows)))
            for lag,value in enumerate(acf[:61]):acfrows.append(dict(L=L,beta=beta,start=start,observable=observable,lag_sweeps=lag,acf=float(value)))
    for L,beta in references:
        for observable in OBS[:3]:
            rows=[r for r in summaries if r["L"]==L and r["beta"]==beta and r["observable"]==observable]
            chains=[np.array([float(r[observable]) for r in groups[L,beta,start] if r["post_burnin"].lower()=="true"]) for start in STARTS]
            pooled.append(dict(L=L,beta=beta,observable=observable,chains=3,retained_samples=sum(r["n"] for r in rows),
                mean=float(np.mean([r["mean"] for r in rows])),MCSE=math.sqrt(sum(r["MCSE"]**2 for r in rows))/3,
                summed_ESS=sum(r["ESS"] for r in rows),min_chain_ESS=min(r["ESS"] for r in rows),
                max_tau_sweeps=max(r["tau_int_sweeps"] for r in rows),split_Rhat=rhat(chains),
                chain_mean_span=max(r["mean"] for r in rows)-min(r["mean"] for r in rows),iid_mean=rows[0]["iid_mean"],iid_SE=rows[0]["iid_SE"],
                mean_acceptance=np.mean([r["mean_acceptance"] for r in rows])))
        for cls in (1,2,3):
            refcount=sum(int(r["class_id"])==cls for r in references[L,beta]);refn=len(references[L,beta])
            count=sum((1 if float(r["n02_fraction"])==0 else 2 if float(r["n02_fraction"])==1 else 3)==cls for start in STARTS for r in groups[L,beta,start] if r["post_burnin"].lower()=="true")
            classrows.append(dict(L=L,beta=beta,class_id=cls,local_count=count,local_n=3*(manifest["sweeps"]-manifest["burnin"]),iid_count=refcount,iid_n=refn,
                iid_fraction=refcount/refn,zero_count_95pct_iid_upper=1-.05**(1/refn) if refcount==0 else "",interpretation="empirical frequency, not exact class probability"))
    write("local_chain_summary.csv",summaries);write("local_pooled_summary.csv",pooled);write("local_acf.csv",acfrows);write("local_empirical_class_probabilities.csv",classrows)
    plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":.2,"font.size":10})
    beta=[float(r["beta"]) for r in exact]
    fig,axes=plt.subplots(2,2,figsize=(11,7.8))
    axes[0,0].plot(beta,[float(r["mean_acceptance"]) for r in exact],"o-");axes[0,0].set(ylabel="Stationary acceptance",xlabel="beta",ylim=(0,1.05),title="Exact L=2: 70 half-filled outcomes")
    for key,label in (("absolute_gap_single_proposal","One proposal"),("gap_one_sweep","One sweep = 8 proposals")):
        axes[0,1].plot(beta,[float(r[key]) for r in exact],"o-",label=label)
    axes[0,1].set(xlabel="beta",ylabel="Absolute spectral gap",title="Exact finite transition matrix");axes[0,1].legend(fontsize=8)
    for cls in (1,2,3):axes[1,0].plot(beta,[float(r[f"p_class{cls}"]) for r in exact],"o-",label=f"Class {cls} (sum over all masks)")
    axes[1,0].set(xlabel="beta",ylabel="Total class probability",yscale="log");axes[1,0].legend(fontsize=8)
    for k,name in enumerate(("uniform_A","AB_checker","02_stripe","mixed")):
        axes[1,1].plot(beta,[float(r["p_"+name]) for r in exact],label=name,**PERIODIC_STYLES[k])
    axes[1,1].set(xlabel="beta",ylabel="One specified suppressed E probability",yscale="log");axes[1,1].legend(fontsize=8)
    fig.suptitle("Exact enumeration: a particular outcome and a whole class have different probabilities")
    fig.tight_layout();save(fig,"local_exact_enumeration")
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for col,L in enumerate((6,12,18)):
        for row,obs in enumerate(("imbalance","n02_fraction")):
            ax=axes[row,col]
            for k,start in enumerate(STARTS):
                rs=[r for r in summaries if r["L"]==L and r["start"]==start and r["observable"]==obs]
                ax.errorbar([r["beta"]+(k-1)*.025 for r in rs],[r["mean"] for r in rs],yerr=[2*r["MCSE"] for r in rs],fmt="o",ms=4,color=COLORS[k],capsize=2,label="initial E: "+start)
            rs=[r for r in pooled if r["L"]==L and r["observable"]==obs]
            ax.errorbar([r["beta"] for r in rs],[r["iid_mean"] for r in rs],yerr=[2*r["iid_SE"] for r in rs],fmt="D--",ms=4,mfc="white",color="black",capsize=2,label="512 independent reference")
            ax.set(title=f"L={L}",xlabel="beta",ylabel=obs)
    axes[0,0].legend(fontsize=7);fig.suptitle("Local-swap chain means vs independent target samples")
    fig.text(.5,.005,"Three starts; 400 sweeps each, first 100 discarded. Error bars: 2 estimated MCSE (local, ACF adjusted) or 2 SE (IID).",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.025,1,.95));save(fig,"local_iid_comparison")
    fig,axes=plt.subplots(2,2,figsize=(11,7.5))
    for col,L in enumerate((6,12,18)):
        rs=[r for r in pooled if r["L"]==L and r["observable"]=="n02_fraction"]
        axes[0,0].plot([r["beta"] for r in rs],[r["mean_acceptance"] for r in rs],label=f"L={L}",**LINE_STYLES[col])
        axes[0,1].plot([r["beta"] for r in rs],[min(q["min_chain_ESS"] for q in pooled if q["L"]==L and q["beta"]==r["beta"]) for r in rs],label=f"L={L}",**LINE_STYLES[col])
        axes[1,0].plot([r["beta"] for r in rs],[max(q["split_Rhat"] or 1 for q in pooled if q["L"]==L and q["beta"]==r["beta"]) for r in rs],label=f"L={L}",**LINE_STYLES[col])
        axes[1,1].plot([r["beta"] for r in rs],[max(abs(q["difference_over_combined_SE"]) for q in summaries if q["L"]==L and q["beta"]==r["beta"] and q["observable"]!="site_memory") for r in rs],label=f"L={L}",**LINE_STYLES[col])
    titles=("Mean acceptance (after burn-in)","Minimum chain ESS across recorded observables","Maximum split R-hat across recorded observables","Maximum local/IID mean difference / combined SE")
    for ax,title in zip(axes.flat,titles):ax.set(xlabel="beta",title=title);ax.legend(fontsize=8)
    fig.suptitle("Finite-run diagnostics: measured observables, not a worst-case mixing bound")
    fig.tight_layout();save(fig,"local_acceptance_ess")
    fig,axes=plt.subplots(3,3,figsize=(13,9))
    for row,beta in enumerate((0.,1.,3.)):
        for col,obs in enumerate(("imbalance","n02_fraction","site_memory")):
            ax=axes[row,col]
            for k,start in enumerate(STARTS):
                rs=groups[18,beta,start];ax.plot([int(r["sweep"]) for r in rs],[float(r[obs]) for r in rs],color=COLORS[k],lw=.6,alpha=.8,label="initial E: "+start)
            ax.axvline(manifest["burnin"],ls="--",color="black",lw=.8);ax.set(title=f"L18, beta={beta:g}: {obs}",xlabel="sweeps")
    axes[0,0].legend(fontsize=8);fig.suptitle("Unthinned trajectories, including burn-in; site memory is relative to each starting mask")
    fig.tight_layout();save(fig,"local_trajectories_L18")
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,obs in zip(axes,("n02_fraction","log_weight")):
        for beta in (0.,.5,1.,2.,3.):
            selected=[r for r in acfrows if r["L"]==18 and r["beta"]==beta and r["observable"]==obs and r["lag_sweeps"]<=30]
            bylag=defaultdict(list)
            for row in selected:bylag[row["lag_sweeps"]].append(row["acf"])
            if len(bylag)>1:ax.plot(sorted(bylag),[np.mean(bylag[lag]) for lag in sorted(bylag)],".-",ms=3,label=f"beta={beta:g}")
        ax.axhline(0,color="black",lw=.5);ax.set(xlabel="lag in sweeps",ylabel="mean within-chain ACF",title="L18: "+obs);ax.legend(fontsize=8)
    fig.suptitle("Post-burn-in ACF averaged across three starts; 300 samples per chain")
    fig.tight_layout();save(fig,"local_acf_L18")
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,L in zip(axes,(6,12,18)):
        for k,kind in enumerate(("uniform_A","AB_checker","02_stripe","02_checker","mixed")):
            rs=[r for r in specific if int(r["L"])==L and r["target_pattern"]==kind]
            ax.plot([float(r["beta"]) for r in rs],[float(r["log10_probability"]) for r in rs],label=kind+f" (d={rs[0]['d_eigenvalues_near_one']})",**PERIODIC_STYLES[k])
        ax.set(title=f"L={L}",xlabel="beta",ylabel="log10 probability of one specified mask")
        ax.legend(fontsize=7)
    fig.suptitle("Specified occupied target S; suppressed E = complement of S")
    fig.text(.5,.005,"d counts eigenvalues of G_E within 1e-9 of one: strong-measurement weight ~ exp(-4 beta d). beta <= 3 need not be asymptotic; these are not total class probabilities.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.95));save(fig,"local_periodic_probabilities")
    fig,axes=plt.subplots(1,5,figsize=(13,3.1))
    for ax,kind in zip(axes,("uniform_A","AB_checker","02_stripe","02_checker","mixed")):
        S=target_mask(2,kind)
        for y in range(2):
            for x in range(2):
                for sub,marker,color in ((0,"o","#286B9C"),(1,"s","#D47728")):
                    i=2*(x+2*y)+sub;ax.plot(x+sub*.27,y,marker,ms=11,mec=color,mew=1.5,mfc=color if i in S else "white")
        ax.set_aspect("equal");ax.set_xlim(-.35,1.65);ax.set_ylim(-.4,1.4);ax.set_title(kind,fontsize=10);ax.set_axis_off()
    fig.suptitle("Periodic target masks S: filled = target occupied; empty = suppressed E")
    fig.text(.5,.04,"Each panel shows a repeated 2 x 2 cell pattern. A = blue circle; B = orange square. Stripes alternate along x; checker alternates along x+y.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.1,1,.9));save(fig,"local_periodic_target_masks")
    entropy_path=ROOT/"analysis/entropy_pipeline_audit.json"
    if entropy_path.exists():
        entropy=json.loads(tracked(entropy_path));assert entropy["status"]=="PASS"
        fig,axes=plt.subplots(1,3,figsize=(12,4))
        quantities=("S2_A","S2_B","paired_A_minus_B")
        for ax,beta in zip(axes,(0.,1.,3.)):
            for index,method in enumerate(("local","iid")):
                row=next(r for r in entropy["results"] if r["L"]==6 and r["beta"]==beta and r["sampler"]==method)
                ax.errorbar(np.arange(3)+(index-.5)*.12,[row[q]["mean"] for q in quantities],
                    yerr=[2*row[q]["block_se"] for q in quantities],fmt="o" if index==0 else "s",ms=8 if index==0 else 5,
                    mfc="white" if index==0 else "#D47728",mec=COLORS[index],color=COLORS[index],capsize=3,label=method)
            ax.set_xticks(range(3),["S2(A)","S2(B)","paired A-B"]);ax.set(title=f"L6, beta={beta:g}",ylabel="Mean Renyi entropy / entropy difference")
        axes[0].legend();fig.suptitle("Same postmeasurement entropy pipeline: local vs independent sampling")
        fig.text(.5,.005,"1000 outcomes per method and beta; errors are 2 block-SE. A-B is a finite-size observable, not an extracted corner coefficient. Small x-offsets separate markers.",ha="center",fontsize=8)
        fig.tight_layout(rect=(0,.04,1,.93));save(fig,"local_entropy_pipeline_comparison")
    tracked(Path(__file__))
    summary=dict(status="PASS",files=FILES,counts=dict(local_chains=len(groups),sweeps_per_chain=manifest["sweeps"],burnin=manifest["burnin"],recorded_trace_rows=len(traces),post_burnin_samples=45*(manifest["sweeps"]-manifest["burnin"]),iid_samples=len(iid),exact_configurations=70),
        conventions=dict(one_sweep="2a randomly selected occupied/empty exchange proposals; sites need not each be selected once",target="p(E)=det[I-(1-exp(-4beta))*G_E]/Za, conditioned on |E|=a",G="Fixed saved LAPACK first-a-eigenvector projector at each L; zero-mode convention is retained",chain_start_names="Patterns define the initial suppressed set E; the occupied target is its complement S",periodic_target_table="In periodic_target_probabilities.csv and target schematic, patterns instead explicitly define occupied target S, E=S complement; retained E-defined raw table is separately named local_periodic_probabilities.csv",memory="2 |E_t intersection E_initial| / a -1; decay is not proof all observables have mixed",ESS="Geyer initial positive, monotone pairs (rho0+rho1),(rho2+rho3),...; g=-1+2 sum(pairs), conservatively clamped g>=1 so negative ACF cannot yield ESS>n. Uses autocovariance denominator n; finite-sample diagnostic only",Rhat="Split 3 chains into 6 segments; classic variance ratio, not rank-normalized",zero_class_count="0 IID hits means not observed, never exact zero; explicit one-sided 95% upper bound included",uncertainty="MCSE and IID SE estimate sampling uncertainty; no universal mixing-time claim"),
        maximum_inverse_drift=max(r["max_inverse_drift"] for r in summaries),maximum_logweight_drift=max(r["max_logweight_drift"] for r in summaries),
        max_log_normalizer_error=max(float(r["normalization_error"]) for r in exact),
        pooled_summary=pooled)
    (DATA/"local_analysis_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(status="PASS",counts=summary["counts"],inverse_drift=summary["maximum_inverse_drift"],logweight_drift=summary["maximum_logweight_drift"]),indent=2))
if __name__=="__main__":main()
