"""Read-only Ref reproduction, with independent SVD fits and explicit snapshots.

Run from any directory with Python + numpy. No simulation except the separately
invoked check_saved_l6.jl. Never imports the primary analysis or DP_Lib.
"""
from pathlib import Path
import csv, hashlib, json, re
from datetime import datetime, timezone
import numpy as np

HERE = Path(__file__).resolve().parent
AI = HERE.parent
REF = AI / "Ref"
BASELINE = AI / "data/ground_observations_baseline.csv"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))

def write_csv(name, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with (HERE/name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)

def fit(x, y, model="log"):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    columns = [np.log(x), np.ones(len(x))]
    if model in ("log_inv", "log_inv2"):
        columns.append(1/x)
    if model == "log_inv2":
        columns.append(1/x**2)
    a = np.column_stack(columns)
    # Column scaling + SVD, independent of the notebook's normal-equation inverse.
    scale = np.linalg.norm(a, axis=0)
    coef = np.linalg.lstsq(a/scale, y, rcond=None)[0] / scale
    err = a@coef-y
    result = dict(model=model,n=len(x),grid=";".join(f"{v:g}" for v in x),
                  lower=float(min(x)),upper=float(max(x)),c=coef[0],b=coef[1],
                  rmse=float(np.sqrt(np.mean(err**2))),max_error=float(max(abs(err))))
    if len(coef)>2: result["inv"] = coef[2]
    if len(coef)>3: result["inv2"] = coef[3]
    return result

def main():
    notebook = json.loads((REF/"neqU=0_pd.ipynb").read_text(encoding="utf-8"))
    snapshots=[]
    with (HERE/"notebook_source_and_text.txt").open("w",encoding="utf-8") as h:
        for i,cell in enumerate(notebook["cells"]):
            h.write(f"\n--- cell {i}; execution_count={cell.get('execution_count')} ---\n")
            h.write("".join(cell.get("source",[])))
            for output in cell.get("outputs",[]):
                for text in (output.get("text",[]),output.get("data",{}).get("text/plain",[])):
                    h.write("\nOUTPUT:\n"+"".join(text))
    ref=np.loadtxt(REF/"neq-eqC_H0.csv",delimiter=",")
    for maximum, skip_header in ((69,True),(72,True),(72,False)):
        v=ref[ref[:,0]<=maximum]
        if skip_header: v=v[1:]
        tag=f"maxL{maximum}_default_header{skip_header}"
        snapshots.append(dict(snapshot=tag,normalization="Ref_double",stage="direct_log_xidx8",**fit(v[8:,0],v[8:,1])))
        xx=[];ss=[]
        for idx in range(5,len(v)-6):
            result=fit(v[idx:,0],v[idx:,1]);xx.append(1/v[idx,0]);ss.append(result["c"])
            snapshots.append(dict(snapshot=tag,normalization="Ref_double",stage="cutoff_log",**result))
        for degree in range(1,6):
            poly=np.polyfit(xx,ss,degree)
            snapshots.append(dict(snapshot=tag,normalization="Ref_double",stage="cutoff_extrapolation",
                model=f"polynomial_degree{degree}",n=len(xx),grid=";".join(f"{1/u:g}" for u in xx),
                c=float(poly[-1]),single_species_c=float(poly[-1]/2),
                rmse=float(np.sqrt(np.mean((np.polyval(poly,xx)-ss)**2)))))
    write_csv("ref_equilibrium_reproduction.csv",snapshots)

    observations=read_csv(BASELINE)
    ua=[r for r in observations if r["pattern"]=="uniform_A" and float(r["beta"])==0]
    eq={int(r["L"]):r for r in ua if np.isinf(float(r["theta"]))}
    common=[]
    for l,value in ref:
        if int(l) in eq:
            row=eq[int(l)];delta=float(row["S_A"])-float(row["S_B"])
            common.append(dict(L=int(l),ref_EC=value,current_D=float(row["D"]),
                ref_minus_twice_D=value-2*float(row["D"]),
                current_delta_minus_D=delta-float(row["D"]),
                current_SA_minus_SC=float(row["S_A"])-float(row["S_C"])))
    write_csv("ref_current_equilibrium_values.csv",common)
    modelrows=[]
    for label, data in (
        ("Ref_half_all_L18_to72",[(l,y/2) for l,y in ref if l>=18]),
        ("Ref_half_common_L18_to72",[(l,y/2) for l,y in ref if l>=18 and int(l) in eq]),
        ("current_common_L18_to72",[(l,float(r['D'])) for l,r in eq.items() if 18<=l<=72]),
        ("current_primary_L18_to90",[(l,float(r['D'])) for l,r in eq.items() if 18<=l<=90]),
    ):
        data=sorted(data)
        for model in ("log","log_inv","log_inv2"):
            modelrows.append(dict(dataset=label,normalization="single",**fit(*zip(*data),model=model)))
    write_csv("equilibrium_model_comparison.csv",modelrows)
    saved_text="".join(notebook["cells"][14]["outputs"][0]["data"]["text/plain"])
    saved={float(t):float(y) for t,y in re.findall(r"^([0-9.]+)\s+([0-9.]+)$",saved_text,re.M)}
    l6=[]
    local=HERE/"current_l6_time_values.csv"
    if local.exists():
        for row in read_csv(local):
            t=float(row["theta"]); value=float(row["twice_D"])
            l6.append(dict(L=6,theta=t,ref_notebook_printed=saved[t],current_twice_D=value,
                           difference=value-saved[t],within_six_decimal_rounding=abs(value-saved[t])<=5.001e-7))
        write_csv("saved_nonequilibrium_L6_comparison.csv",l6)

    # These are transcribed from saved, hashed notebook figures, not reproduced fits.
    saved_figures=[
        dict(cell=4,description="eq direct log, saved L<=69 snapshot",Ref_c=.343,single_c=.343/2),
        dict(cell=6,description="eq fifth-order cutoff extrapolation",Ref_c=.3162,single_c=.3162/2),
        dict(cell=8,description="neq direct log, source table missing",Ref_c=.375,single_c=.375/2),
        dict(cell=13,description="neq second-order cutoff extrapolation",Ref_c=.330,single_c=.330/2),
        dict(cell=13,description="neq third-order cutoff extrapolation",Ref_c=.328,single_c=.328/2),
        dict(cell=13,description="neq fourth-order cutoff extrapolation",Ref_c=.314,single_c=.314/2),
        dict(cell=13,description="neq nonlinear power cutoff extrapolation",Ref_c=.310,single_c=.310/2),
    ]
    write_csv("saved_figure_coefficients.csv",saved_figures)
    sources=[*REF.glob("*"),BASELINE,
             AI/"fermion_experiment.jl",Path("D:/JuliaDQMC/code/DP_Lib.py"),Path(__file__),HERE/"check_saved_l6.jl"]
    if local.exists():sources.append(local)
    summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        hashes={str(p.resolve()):sha(p) for p in sources if p.is_file()},
        normalization="Ref -log(abs2(det(GM))) = twice one-species S2; Ref SA-SB equals 2D when SA=SC",
        common_eq_rows=len(common),max_abs_ref_minus_twice_D=max(abs(r["ref_minus_twice_D"]) for r in common),
        max_abs_current_SA_minus_SC=max(abs(r["current_SA_minus_SC"]) for r in common),
        saved_eq_snapshot="L<=69, default pandas header drops L=3; xidx1=8 => L30:3:69; 11 cutoff fits",
        current_csv_snapshot="L<=72; same code now yields 12 cutoff fits and a different extrapolation",
        saved_neq_direct_vs_saved_eq_direct_relative_gap=.375/.343-1,
        saved_neq_extrapolation_range_single=[.310/2,.330/2],
        saved_eq_extrapolation_single=.3162/2,
        missing_nonequilibrium_tables=["neqU0EE_SC.csv","neqU0EE_Half - <original suffix>.csv","neqU0EE_HalfHalf - <original suffix>.csv"],
        limitations=[
          "Saved plots/code/data do not share a consistent execution snapshot; exact neq coefficient reproduction is unavailable without missing CSVs.",
          "Located DP_Lib.linear_fit takes mandatory dy but notebook calls two arguments, and cell13 assumes different return structure.",
          "Located unit_neardata with zero uncertainties chooses most frequent EXACT equal values, last group on ties; this is not a numerical size-convergence criterion.",
          "Ref B CSV mixes row-oriented records and one-value-per-line records; C is the well-formed fit input.",
          "Fifth-degree extrapolation of strongly correlated nested-cutoff slopes is model-sensitive and not an independent precision estimate.",
          "Saved figure8 and figure13 depend on unavailable earlier kernel state; only labels, not underlying observations, are recoverable.",
        ],
        l6_rows=len(l6),
        l6_max_abs_error_through_5_5=max((abs(r["difference"]) for r in l6 if r["theta"]<=5.5),default=None),
        l6_max_abs_error_all=max((abs(r["difference"]) for r in l6),default=None),
        time_coordinate_conclusion="L6 matches current theta with no rescaling through theta5.5 within notebook six-decimal rounding; last old values have ~2e-5 drift.",
        recommended_method="Choose a size-converged regime from independent size comparison; compare pure-log and explicit correction models on declared windows. Do not select windows or model order by agreement with an equilibrium target.")
    (HERE/"reference_audit_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k not in ('hashes','limitations')},indent=2))

if __name__=="__main__":main()
