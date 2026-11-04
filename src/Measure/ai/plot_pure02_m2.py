"""Plot the blocked 2x2 pure-0/2 audit and the seven reachable examples.

Reads frozen numerical CSV/JSON files only. No entropy or projection calculation
is performed. All new figures and a hash receipt stay in figures_pure02_m2/.
"""
from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import plot_occupancy_classes_patterns as schematic
import analyze_occupancy_classes as old
from analyze_occupancy_classes import core, previous

HERE = Path(__file__).resolve().parent
OUT = HERE/"figures_pure02_m2"
SOURCES = {}
CLASS_LABELS = {1:"Class 1 | One particle in every cell",
                2:"Class 2 | Empty and doubly occupied cells only",
                3:"Class 3 | Empty, single and double cells coexist"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    value=path.read_bytes();SOURCES[path]=sha(value)
    return value


def read_json(path):
    return json.loads(read(path).decode("utf-8-sig"))


def read_csv(path):
    return list(csv.DictReader(read(path).decode("utf-8-sig").splitlines()))


def verify(path, expected):
    if sha(read(path)) != expected:
        raise ValueError(f"Source hash mismatch: {path}")


def istrue(value):
    return str(value).lower()=="true"


def numeric_coefficients(rows):
    result=[]
    for row in rows:
        parsed={}
        for key,value in row.items():
            if value=="": parsed[key]=None
            elif value.lower() in ("true","false"):parsed[key]=istrue(value)
            else:
                try:parsed[key]=float(value)
                except ValueError:parsed[key]=value
        result.append(parsed)
    return result


def input_data():
    audit=read_json(HERE/"data/pure02_m2_summary.json")
    if audit.get("status")!="PASS":raise ValueError("2x2 audit is not PASS")
    for path,expected in audit["files"].items():
        verify(HERE/path,expected)
    reach=read_csv(HERE/"data/pure02_m2_reachability.csv")
    projection=read_csv(HERE/"data/pure02_m2_projection.csv")
    expected_names=[f"pure02_m2_cells_{pair}" for pair in ("12","13","14","23","24","34")]
    if set(r["pattern"] for r in reach)!=set(expected_names):raise ValueError("Need all six 2x2 patterns")
    blocked=[]
    for name in expected_names:
        block=[r for r in reach if r["pattern"]==name]
        bits=block[0]["occupied_bits"]
        counts=Counter(int(bits[i])+int(bits[i+1]) for i in range(0,8,2))
        if len(bits)!=8 or bits.count("1")!=4 or counts != {0:2,2:2}:raise ValueError("Incorrect 2x2 pure-0/2 mask")
        if any(r["occupied_bits"]!=bits or int(r["gamma_rank"])!=3 or int(r["negative_dimension"])!=4
               or istrue(r["reachable"]) or float(r["accessible_excess"])<=0 for r in block):
            raise ValueError("Blocked 2x2 classification or rank inconsistency")
        blocked.append(dict(pattern=name,m=2,occupied_bits=bits,cell_counts=[2,0,2],class_id=2,
                            checked_sizes=sorted(int(r["L"]) for r in block)))
    if set(r["pattern"] for r in projection)!=set(expected_names):raise ValueError("Projection audit is incomplete")
    bits_by_name={p["pattern"]:p["occupied_bits"] for p in blocked}
    for r in projection:
        if r["occupied_bits"]!=bits_by_name[r["pattern"]]:raise ValueError("Projection mask mismatch")
        if r["method"] not in ("strict_spectral_support","direct_qr"):
            raise ValueError(f"Unknown projection method: {r['method']}")
        if math.isinf(float(r["theta"])) and r["method"]!="strict_spectral_support":
            raise ValueError("Finite-time QR must not be marked infinite time")
        if abs(float(r["energy"])-float(r["ground_energy"])-float(r["excess_energy"]))>1e-8:
            raise ValueError("Energy difference identity failed")
    base_plan=read_json(HERE/"analysis_occupancy_classes/plan.json")
    follow_plan=read_json(HERE/"analysis_irregular_followup/plan.json")
    names=list(base_plan["patterns"])+["near02_irregular3"]
    metadata={**base_plan["pattern_metadata"],"near02_irregular3":follow_plan["pattern_metadata"]["near02_irregular3"]}
    ground_rows=read_csv(HERE/"data/occupancy_classes_reachability.csv")+read_csv(HERE/"data/irregular_m3_reachability.csv")
    ground=[]
    for name in names:
        meta=metadata[name];bits=meta["occupied_bits"]
        counts=Counter(int(bits[i])+int(bits[i+1]) for i in range(0,len(bits),2))
        checks=[r for r in ground_rows if r["pattern"]==name]
        if not checks or any(r["occupied_bits"]!=bits or not istrue(r["ground_reachable"])
            or abs(float(r["excess_energy"]))>=float(r["ground_tol"]) for r in checks):
            raise ValueError(f"Reachable-example check failed: {name}")
        pattern_file=HERE/("irregular_m3_patterns.jl" if name=="near02_irregular3" else "occupancy_classes_patterns.jl")
        for recorded in {r["pattern_sha256"] for r in checks}:verify(pattern_file,recorded)
        for recorded in {r["kernel_sha256"] for r in checks}:verify(HERE/"fermion_experiment.jl",recorded)
        ground.append(dict(pattern=name,m=meta["m"],occupied_bits=bits,class_id=meta["occupancy_class"],
            cell_counts=[counts[0],counts[1],counts[2]],checked_sizes=sorted(int(r["L"]) for r in checks),equilibrium=meta["equilibrium"]))
    if Counter(p["class_id"] for p in ground)!={1:2,2:2,3:3}:raise ValueError("Seven-example class counts differ")
    coefficients=read_csv(HERE/"analysis_occupancy_classes/primary_coefficients.csv")
    coefficients += [r for r in read_csv(HERE/"analysis_irregular_followup/primary_coefficients.csv") if r["pattern"]=="near02_irregular3"]
    if len(coefficients)!=63 or len({(r["pattern"],float(r["beta"])) for r in coefficients})!=63:
        raise ValueError("Need 63 unique pattern/beta coefficients")
    for name in names:
        group=[r for r in coefficients if r["pattern"]==name]
        if {float(r["beta"]) for r in group}!={i/4 for i in range(9)}:raise ValueError("Incomplete beta grid")
        config=metadata[name]["equilibrium"]
        if any(int(r["eq_Lmin"])!=config["Lmin"] or int(r["eq_Lmax"])!=config["Lmax"] for r in group):
            raise ValueError("Coefficient window does not match the pattern")
    return audit,blocked,reach,projection,ground,numeric_coefficients(coefficients)


def legend_handles():
    return [Line2D([],[],marker="o" if sub==0 else "s",linestyle="none",ms=7,
            mfc=schematic.COLORS[sub] if occupied else "white",mec=schematic.COLORS[sub],
            label=f"{'AB'[sub]} {'occupied' if occupied else 'empty'}")
            for sub in (0,1) for occupied in (True,False)]


def save(fig,stem,files):
    for ext in ("png","pdf"):
        path=OUT/f"{stem}.{ext}"
        fig.savefig(path,dpi=180,bbox_inches="tight",facecolor="white")
        files.append(path)
    plt.close(fig)


def blocked_catalogue(patterns,files):
    fig,axes=plt.subplots(2,3,figsize=(12.5,9.5))
    fig.subplots_adjust(left=.055,right=.98,bottom=.18,top=.79,hspace=.75,wspace=.25)
    for ax,p in zip(axes.flat,patterns):
        display=dict(p,pattern=p["pattern"].replace("pure02_m2_cells_","Full cells "))
        schematic.draw_panel(ax,display)
        ax.text(.5,-.34,r"Class 2: BLOCKED; $\Gamma$ rank $3/4$",transform=ax.transAxes,ha="center",fontsize=11,color="#7B3F37")
    fig.suptitle("All six half-filled pure-0/2 patterns in a 2 x 2 tile",y=.99,fontsize=16)
    fig.legend(handles=legend_handles(),loc="upper center",bbox_to_anchor=(.5,.95),ncol=4,frameon=False)
    fig.text(.5,.022,"Cell order: 1=(1,1), 2=(2,1), 3=(1,2), 4=(2,2). Two full cells, two empty cells.\n"
             "All six fail the required negative-energy rank at Gamma; none is a ground-state scaling example.",ha="center",fontsize=10)
    save(fig,"pattern_catalogue",files)


def ground_catalogue(patterns,files):
    fig=plt.figure(figsize=(13.5,14.2))
    fig.suptitle("Seven periodic configurations, grouped by cell occupation",y=.99,fontsize=17)
    fig.legend(handles=legend_handles(),loc="upper center",bbox_to_anchor=(.5,.95),ncol=4,frameon=False,fontsize=11)
    for row,category in enumerate((1,2,3)):
        bottom=.655-.29*row
        fig.text(.04,bottom+.245,CLASS_LABELS[category],fontsize=14,fontweight="bold")
        members=[p for p in patterns if p["class_id"]==category]
        for col in range(3):
            ax=fig.add_axes((.04+.32*col,bottom,.28,.185))
            if col>=len(members):ax.set_axis_off();continue
            schematic.draw_panel(ax,members[col])
    fig.text(.5,.018,"Filled = occupied; hollow = empty; grey box = one A/B cell. Cell-coordinate schematic.\n"
             "Ground-state reachable at checked sizes only. Classification does not certify entropy scaling.",ha="center",fontsize=10.5)
    save(fig,"classified_ground_patterns",files)
    p=next(p for p in patterns if p["pattern"]=="near02_irregular3")
    fig,ax=plt.subplots(figsize=(5.6,6.0))
    fig.subplots_adjust(left=.16,right=.94,bottom=.20,top=.71)
    schematic.draw_panel(ax,p)
    fig.legend(handles=legend_handles(),loc="upper center",bbox_to_anchor=(.5,.98),ncol=2,frameon=False)
    fig.text(.5,.04,"Class 3: 4 empty + 1 A-only + 4 double cells.\nReachable at checked sizes; no scaling guarantee.",ha="center",fontsize=11)
    save(fig,"near02_irregular3_pattern",files)


def coefficients_catalogue(patterns,coefficients,files):
    fig=plt.figure(figsize=(15,14))
    fig.suptitle("Seven reachable examples: finite-window equilibrium / time coefficients",y=.99,fontsize=15)
    fig.text(.5,.953,"Independent y scales. Large blue open circles: equilibrium. Small orange squares: time; filled = local pass, open = unresolved.\n"
             "Overview omits sensitivity bands; individual result figures retain them. No 2 x 2 blocked pattern is included.",ha="center",fontsize=10)
    fig.legend(handles=[
        Line2D([],[],color=core.BLUE,marker="o",ms=8,mfc="white",mew=1.4,label="Equilibrium"),
        Line2D([],[],color=core.ORANGE,marker="s",ms=5,mfc=core.ORANGE,ls="--",label="Time: local pass"),
        Line2D([],[],color=core.ORANGE,marker="s",ms=5,mfc="white",ls="--",label="Time: unresolved")],
        loc="upper center",bbox_to_anchor=(.5,.939),ncol=3,frameon=False,fontsize=10)
    for row,category in enumerate((1,2,3)):
        bottom=.655-.29*row
        fig.text(.045,bottom+.245,CLASS_LABELS[category],fontsize=13,fontweight="bold")
        members=[p for p in patterns if p["class_id"]==category]
        for col in range(3):
            left=.05+.32*col
            if col>=len(members):
                ax=fig.add_axes((left,bottom,.275,.19));ax.set_axis_off();continue
            p=members[col];block=[r for r in coefficients if r["pattern"]==p["pattern"]]
            zoom=old.needs_equilibrium_zoom(block)
            ax=fig.add_axes((left,bottom+(.077 if zoom else 0),.275,.125 if zoom else .202))
            low,high=p["equilibrium"]["Lmin"],p["equilibrium"]["Lmax"]
            previous.coefficient_axis(ax,block,f"{p['pattern']}\nEq L{low}..{high}; time L180, theta4..7",show_bands=False)
            # No invisible model/window range enters this overview's limits.
            values=np.concatenate([core.array(block,key) for key in ("c_eq","c_neq")]);values=values[np.isfinite(values)]
            ymin,ymax=min(0.,float(min(values))),max(0.,float(max(values)))
            padding=.07*max(ymax-ymin,.01);ax.set_ylim(ymin-padding,ymax+padding)
            ax.title.set_fontsize(10);ax.tick_params(labelsize=8);ax.set_ylabel("Finite-window c",fontsize=9)
            ax.set_xlabel(r"$\beta$",fontsize=10)
            if ax.get_legend() is not None:ax.get_legend().remove()
            if zoom:
                ax.set_xlabel("");ax.tick_params(labelbottom=False)
                inset=fig.add_axes((left,bottom,.275,.053))
                old.equilibrium_zoom_axis(inset,block,show_bands=False)
                inset.set_title("Equilibrium zoom (independent scale)",fontsize=8)
                inset.set_ylabel("Eq c",fontsize=8);inset.tick_params(labelsize=7)
                inset.set_xlabel(r"$\beta$",fontsize=9)
    fig.text(.5,.02,"A fitted coefficient remains unresolved when time checks fail. Local pass is not an asymptotic proof.\n"
             "Original six patterns + near02_irregular3, nine beta values each (63 distinct pattern-beta pairs).",ha="center",fontsize=10)
    save(fig,"classified_ground_coefficients",files)


def energy_projection(patterns,rows,files):
    sizes=sorted({int(r["L"]) for r in rows})
    for size in sizes:
        fig,axes=plt.subplots(2,3,figsize=(13,8.7))
        fig.subplots_adjust(left=.07,right=.98,bottom=.13,top=.84,wspace=.27,hspace=.35)
        for ax,p in zip(axes.flat,patterns):
            block=[r for r in rows if r["pattern"]==p["pattern"] and int(r["L"])==size]
            for method,color,marker,style,label in (("strict_spectral_support",core.BLUE,"o","-","Exact-support reference"),
                ("direct_qr",core.ORANGE,"s","--","Direct Float64 QR")):
                points=sorted((r for r in block if r["method"]==method and math.isfinite(float(r["theta"]))),key=lambda r:float(r["theta"]))
                if points:
                    ax.plot([float(r["theta"]) for r in points],[float(r["excess_energy"]) for r in points],
                        marker=marker,linestyle=style,color=color,ms=4,mfc="white",lw=1.4,label=label)
            limits=[float(r["excess_energy"]) for r in block if r["method"]=="strict_spectral_support" and math.isinf(float(r["theta"]))]
            if len(limits)!=1:raise ValueError(f"Missing unique accessible limit for {p['pattern']} L{size}")
            ax.axhline(limits[0],color=".3",ls=":",lw=1.1,label="Exact accessible limit")
            ax.set(title=p["pattern"].replace("pure02_m2_cells_","Full cells "),xlabel=r"$\theta$",ylabel=r"Total $E(\theta)-E_0$")
            ax.set_xscale("symlog",linthresh=1)
            ax.set_xlim(-.1,170)
            ax.set_xticks([0,1,4,16,128]);ax.set_xticklabels(["0","1","4","16","128"])
            ax.axhline(0,color=".75",lw=.7);ax.grid(axis="y",alpha=.22)
            ax.legend(fontsize=7)
        fig.suptitle(f"Imaginary-time projection energy: L={size}, beta=0\nExact initial spectral support versus unrestricted floating-point QR",fontsize=13,y=.98)
        fig.text(.5,.025,"Energy diagnostic, not entanglement scaling; no corner coefficient is fitted. Theta axis: linear near zero, logarithmic above 1.\n"
            "A floating-point QR trajectory can leak into exactly forbidden sectors; any such late-time drift is retained.",ha="center",fontsize=10)
        save(fig,"projection_energy" if len(sizes)==1 else f"projection_energy_L{size}",files)


def energy_sizes(patterns,rows,files):
    fig,axes=plt.subplots(2,3,figsize=(13,8.5))
    fig.subplots_adjust(left=.07,right=.98,bottom=.12,top=.85,wspace=.27,hspace=.33)
    for ax,p in zip(axes.flat,patterns):
        block=sorted((r for r in rows if r["pattern"]==p["pattern"]),key=lambda r:int(r["L"]))
        ax.plot([int(r["L"]) for r in block],[float(r["accessible_excess"]) for r in block],"o-",color=core.BLUE,mfc="white")
        ax.set(title=p["pattern"].replace("pure02_m2_cells_","Full cells "),xlabel="L",ylabel=r"Total $E_{\rm acc}-E_0$")
        ax.axhline(0,color=".5",lw=.7);ax.grid(axis="y",alpha=.22)
    fig.suptitle("Exact-support projection limit: positive energy excess across checked sizes\nEach panel is one of the six pure-0/2 2 x 2 configurations",fontsize=13,y=.98)
    fig.text(.5,.025,"Total energy excess, not entropy and not an entropy scaling fit. Independent y scales; lines guide the eye.\n"
             "Nonzero excess certifies failure to reach a ground state at that size; Gamma rank deficiency supplies the analytical obstruction.",ha="center",fontsize=10)
    save(fig,"energy_excess_vs_L",files)


def main():
    audit,blocked,reach,projection,ground,coefficients=input_data()
    OUT.mkdir(exist_ok=True)
    for module in (schematic,old,core,previous):read(Path(module.__file__).resolve())
    read(Path(__file__).resolve())
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
    files=[]
    blocked_catalogue(blocked,files)
    ground_catalogue(ground,files)
    coefficients_catalogue(ground,coefficients,files)
    energy_projection(blocked,projection,files)
    energy_sizes(blocked,reach,files)
    for path,expected in SOURCES.items():
        if sha(path.read_bytes())!=expected:raise ValueError(f"Input changed during plotting: {path}")
    receipt=dict(status="PASS",generated_utc=datetime.now(timezone.utc).isoformat(),
        scope="Plotting only: frozen source values and schematics. No new projection, entropy calculation or fit.",
        source_sha256={str(p.relative_to(HERE)).replace('\\','/'):v for p,v in SOURCES.items()},
        files={**{str(p.relative_to(HERE)).replace('\\','/'):v for p,v in SOURCES.items()},
               **{str(p.relative_to(HERE)).replace('\\','/'):sha(p.read_bytes()) for p in files}},
        blocked_patterns=blocked,ground_examples=ground,coefficient_pairs=len(coefficients),
        projection_methods=sorted({r["method"] for r in projection}),
        interpretation="All 2x2 pure-0/2 examples are blocked; never included in the 63 reachable-example coefficient pairs.")
    (OUT/"receipt.json").write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(dict(status="PASS",figure_pairs=len(files)//2,blocked_patterns=len(blocked),ground_examples=len(ground),coefficient_pairs=len(coefficients)),indent=2))


if __name__=="__main__":main()
