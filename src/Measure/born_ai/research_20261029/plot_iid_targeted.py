"""Source-verified figures for targeted identities and bounds."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ANALYSIS=HERE/"iid_targeted_analysis"
OUT=HERE/"figures"


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with path.open(newline="") as stream:return list(csv.DictReader(stream))


def main():
    summary_path=ANALYSIS/"summary.json"
    summary=json.loads(summary_path.read_text())
    assert summary["status"]=="PASS"
    sources={str(summary_path.relative_to(HERE)):sha(summary_path),Path(__file__).name:sha(Path(__file__))}
    for filename in ("means.csv","bounds.csv"):
        p=ANALYSIS/filename;relative=str(p.relative_to(HERE))
        assert sha(p)==summary["outputs_sha256"][relative]
        sources[relative]=sha(p)
    means=read(ANALYSIS/"means.csv");bounds=read(ANALYSIS/"bounds.csv")
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
        "axes.grid":True,"grid.alpha":.2,"savefig.dpi":180,"pdf.fonttype":42})
    colors={1.:"#246A9B",3.:"#CE742B"}
    outputs=[]
    def save(fig,name):
        for ext in ("png","pdf"):
            p=OUT/f"{name}.{ext}"
            if p.exists():raise FileExistsError(p)
            fig.savefig(p,bbox_inches="tight");outputs.append(p)
        plt.close(fig)
    def row(L,beta,name):
        return next(r for r in means if int(r["L"])==L and float(r["beta"])==beta and r["observable"]==name)

    fig,axes=plt.subplots(1,2,figsize=(12.8,4.5),constrained_layout=True)
    for beta,offset,marker in ((1.,-.12,"o"),(3.,.12,"s")):
        vals=[float(row(L,beta,"channel_residual")["z_mean_over_reported_SE"]) for L in (6,12,18)]
        axes[0].errorbar(np.arange(3)+offset,vals,yerr=np.ones(3),fmt=marker,mfc="white" if beta==1 else colors[beta],
            color=colors[beta],capsize=4,label=fr"$\beta={beta:g}$",ms=6)
    axes[0].set(xticks=range(3),xticklabels=[6,12,18],xlabel="L",ylabel=r"$(\widehat\kappa-\kappa)/\mathrm{MCSE}$",title="Exact channel factor: one linear projection")
    axes[0].legend()
    parameters=[(L,beta) for L in (6,12,18) for beta in (1.,3.)]
    for region,offset,color,marker in (("A",-.2,"#246A9B","o"),("B",0.,"#CE742B","s"),("C",.2,"#55934D","^")):
        vals=[float(row(L,beta,f"Vlaw_{region}_residual")["z_mean_over_reported_SE"]) for L,beta in parameters]
        axes[1].errorbar(np.arange(6)+offset,vals,yerr=np.ones(6),fmt=marker,mfc="white",color=color,capsize=3,label=region,ms=5)
    axes[1].set(xticks=range(6),xticklabels=[f"{L}, {int(beta)}" for L,beta in parameters],xlabel=r"$(L,\beta)$",
        ylabel="Variance-decomposition residual / MCSE",title=r"$\langle V_R+(\langle N_R\rangle_E-N_{R,0})^2\rangle-V_{R,0}$")
    axes[1].legend(ncol=3)
    for ax in axes:
        ax.axhline(0,color="#333333",lw=1)
        ax.axhline(2,color="#777777",ls=":",lw=1);ax.axhline(-2,color="#777777",ls=":",lw=1)
        ax.set_ylim(-3.5,3.8)
    fig.suptitle("2,048 independent states per parameter; four batches\n1 MCSE = max(iid SE, four-batch SE); error bars are not strict confidence intervals.",fontsize=11)
    save(fig,"iid_identity_checks")

    fig,axes=plt.subplots(2,2,figsize=(11.8,8),constrained_layout=True)
    for col,beta in enumerate((1.,3.)):
        for ri,observable in enumerate(("SA","DV_full")):
            ax=axes[ri,col]
            selected=[r for r in bounds if float(r["beta"])==beta and r["observable"]==observable]
            selected.sort(key=lambda r:int(r["L"]))
            L=np.array([int(r["L"]) for r in selected])
            mean=np.array([float(r["mean"]) for r in selected]);error=np.array([float(r["MCSE_reported"]) for r in selected])
            lo=np.array([float(r["deterministic_lower_bound"]) for r in selected])
            ax.errorbar(L,mean,yerr=error,color=colors[beta],marker="o",mfc="white",ms=7,capsize=4,label="Born mean ± 1 MCSE")
            ax.plot(L,lo,color="#55934D",ls="--",marker="v",mfc="white",label="Exact lower bound")
            if observable=="SA":
                hi=np.array([float(r["deterministic_upper_bound"]) for r in selected])
                ax.plot(L,hi,color="#777777",ls=":",marker="^",mfc="white",label="Exact upper bound")
                ax.set_ylabel(r"$\langle S_2(A)\rangle$ and bounds")
            else:ax.set_ylabel(r"$\langle D_V^{\rm full}\rangle$ and lower bound")
            ax.set(xlabel="L",xticks=[6,12,18],yscale="log",title=fr"$\beta={beta:g}$")
            ax.legend(fontsize=8.5)
    fig.suptitle("Exact inequalities; lines guide the eye, no scaling fit\n"
        r"Bounds: $2\kappa^2V_{A,0}$, $4\ln(2)V_{A,0}$; $\kappa^2D_{V,0}^{\rm full}$. "
        "1 MCSE = max(iid SE, four-batch SE), not a strict CI.",fontsize=10)
    save(fig,"iid_bounds_checks")

    (OUT/"plot_receipt.json").write_text(json.dumps(dict(role="Plots of verified saved summaries; no resampling or refitting",inputs_sha256=sources,
        outputs_sha256={str(p.relative_to(HERE)):sha(p) for p in outputs}),indent=2)+"\n")
    print("SAVED",len(outputs),"figure files")


if __name__=="__main__":main()
