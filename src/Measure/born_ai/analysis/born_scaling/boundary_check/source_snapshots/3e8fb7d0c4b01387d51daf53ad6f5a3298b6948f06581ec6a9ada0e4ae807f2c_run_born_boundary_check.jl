# Paired geometry / translation diagnostics; all historical production is read-only.
include("run_born_scaling.jl")
include("fused_original_swap_kernel.jl")

const BOUNDARY_COLUMNS=("L","beta","mu","chain","seed","start","sweep","record_index",
    "measure_every","p_num","p_den","tx","ty","translation_index","SA","SB","SC","D",
    "accepted","proposed","log_weight","stage")

function boundary_indices(L,w,h,tx,ty)
    sort!([2(mod(x+tx,L)+mod(y+ty,L)*L)+s for y in 0:h-1 for x in 0:w-1 for s in 1:2])
end

function canonical_region(idx,N)
    other=setdiff(1:N,idx)
    length(idx)>length(other) && return other
    # Pure-state complement identity permits a cache for translated half-system strips.
    length(idx)==length(other) && Tuple(other)<Tuple(idx) && return other
    idx
end

struct BoundaryWorkspace
    Q::Matrix{Float64}
    tau::Vector{Float64}
    indices::Vector{Vector{Int}}
    entries::Vector{NamedTuple}
    V::Matrix{Float64}
    C::Matrix{Float64}
    B::Matrix{Float64}
end

function boundary_workspace(model;full_translations=false)
    L=model.L; indices=Vector{Int}[]; entries=NamedTuple[]
    cache=Dict{Tuple,Int}()
    function index_id(idx)
        idx=canonical_region(idx,model.N); key=Tuple(idx)
        get!(cache,key) do
            push!(indices,idx); length(indices)
        end
    end
    for m in (3,2)
        L%m==0 || error("geometry needs an integer width")
        w=L÷m
        shifts=full_translations ? collect(0:L-1) : collect(0:w:L-1)
        tid=0
        for ty in shifts,tx in shifts
            tid+=1
            ids=(index_id(boundary_indices(L,w,L,tx,ty)),
                 index_id(boundary_indices(L,w,L-w,tx,ty)),
                 index_id(boundary_indices(L,L,w,tx,ty)))
            push!(entries,(;m,tx,ty,tid,ids))
        end
    end
    kmax=maximum(length,indices)
    BoundaryWorkspace(similar(model.Q),zeros(model.n),indices,entries,
        zeros(kmax,model.n),zeros(kmax,kmax),zeros(kmax,kmax))
end

function boundary_entropies!(work,model,E,beta)
    work.Q.=model.Q
    attenuation=exp(-2beta)
    @inbounds for j in 1:model.n,i in E;work.Q[i,j]*=attenuation;end
    LAPACK.geqrf!(work.Q,work.tau)
    LAPACK.orgqr!(work.Q,work.tau,model.n)
    values=zeros(length(work.indices))
    for (ri,idx) in enumerate(work.indices)
        k=length(idx); V=view(work.V,1:k,:); C=view(work.C,1:k,1:k); B=view(work.B,1:k,1:k)
        @inbounds for j in 1:model.n,i in 1:k;V[i,j]=work.Q[idx[i],j];end
        mul!(C,V,V')
        @inbounds for i in 1:k;C[i,i]-=.5;end
        BLAS.syrk!('U','N',2.,C,0.,B)
        @inbounds for i in 1:k;B[i,i]+=.5;end
        factor=cholesky!(Symmetric(B,:U))
        values[ri]=-2sum(log,diag(factor.U))
    end
    map(work.entries) do entry
        SA,SB,SC=values[collect(entry.ids)]
        D=((entry.m-1)*SA+SC)/entry.m-SB
        (;entry...,SA,SB,SC,D)
    end
end

function boundary_validate(plan)
    plan["state"]=="staggered" && plan["mu"]==1e-8 || error("state mismatch")
    for (file,digest) in plan["source_sha256"]
        sha(joinpath(ROOT,file))==digest || error("source changed: $file")
    end
    BLAS.get_num_threads()==1 || error("single BLAS thread required")
end

function boundary_run(job,plan,planpath)
    samples=joinpath(ROOT,job["sample_path"])
    receipt_path=joinpath(ROOT,job["receipt_path"])
    snapshots=joinpath(ROOT,job["snapshot_path"])
    any(isfile,(samples,receipt_path,snapshots)) && error("refuse to overwrite a diagnostic chain")
    mkpath(dirname(samples))
    model=load_model(job);rng=MersenneTwister(job["seed"])
    E=initial_set(model.L,job["start"],rng);state=swap_state(model.G,E,job["beta"])
    work=boundary_workspace(model);oldwork=EntropyWorkspace(model)
    check=boundary_entropies!(work,model,state.E,job["beta"])
    origin=only(filter(x->x.m==3 && x.tx==0 && x.ty==0,check))
    old=entropy!(oldwork,model,state.E,job["beta"])
    checkerror=maximum(abs.([origin.SA-old.SA,origin.SB-old.SB,origin.SC-old.SC,origin.D-old.D]))
    checkerror<1e-8 || error("legacy entropy check failed")
    checkfile=samples*".initial.csv"
    open(checkfile,"w") do io
        println(io,"p_den,translation_index,tx,ty,SA,SB,SC,D")
        for r in check;csv(io,(r.m,r.tid,r.tx,r.ty,r.SA,r.SB,r.SC,r.D));end
    end
    started=time()
    receipt=Dict{String,Any}("status"=>"running","job"=>job,"plan_sha256"=>sha(planpath),
        "source_sha256"=>plan["source_sha256"],"started_at"=>string(now()),"initial_E"=>copy(E),
        "legacy_entropy_max_difference"=>checkerror,"unique_regions"=>length(work.indices),
        "rows_per_measurement"=>length(work.entries),"snapshot_format"=>"Int32 native little endian, n=L^2 indices per recorded sweep, original E order", "julia_version"=>string(VERSION))
    write_toml(receipt_path,receipt)
    maxdrift=0.;maxlog=0.;accepted=0;proposed=0;wa=0;wp=0;recorded=0
    update_seconds=0.;entropy_seconds=0.;refresh_seconds=0.
    try
        open(samples,"w") do io
            open(snapshots,"w") do binary
                println(io,join(BOUNDARY_COLUMNS,','))
                for sweep in 1:job["sweeps"]
                    result=nothing
                    update_seconds+=@elapsed result=FusedOriginalSwapKernel.sweep_fused!(state,model.G,rng)
                    accepted+=result.accepted;proposed+=result.proposed;wa+=result.accepted;wp+=result.proposed
                    if sweep%job["refresh_every"]==0
                        drift=nothing;refresh_seconds+=@elapsed drift=rebuild!(state,model.G)
                        maxdrift=max(maxdrift,drift.inverse_drift);maxlog=max(maxlog,drift.logweight_drift)
                        maxdrift<1e-5 && maxlog<1e-5 || error("numerical drift threshold exceeded")
                    end
                    if sweep%job["measure_every"]==0
                        rows=nothing;entropy_seconds+=@elapsed rows=boundary_entropies!(work,model,state.E,job["beta"])
                        recorded+=1
                        for r in rows
                            csv(io,(model.L,job["beta"],model.mu,job["chain"],job["seed"],job["start"],sweep,recorded,
                                job["measure_every"],1,r.m,r.tx,r.ty,r.tid,r.SA,r.SB,r.SC,r.D,wa,wp,state.log_weight,job["stage"]))
                        end
                        write(binary,Int32.(state.E));wa=0;wp=0
                    end
                    if sweep%128==0
                        flush(io);flush(binary)
                        println("PROGRESS $(job["run_id"]) sweep=$sweep/$(job["sweeps"]) elapsed=$(round(time()-started,digits=1))")
                        flush(stdout)
                    end
                end
            end
        end
        drift=rebuild!(state,model.G)
        maxdrift=max(maxdrift,drift.inverse_drift);maxlog=max(maxlog,drift.logweight_drift)
        maxdrift<1e-5 && maxlog<1e-5 || error("final drift threshold exceeded")
        receipt["status"]="complete";receipt["completed_at"]=string(now())
        receipt["elapsed_seconds"]=time()-started;receipt["update_seconds"]=update_seconds
        receipt["entropy_seconds"]=entropy_seconds;receipt["refresh_seconds"]=refresh_seconds
        receipt["maximum_inverse_drift"]=maxdrift;receipt["maximum_logweight_drift"]=maxlog
        receipt["accepted"]=accepted;receipt["proposed"]=proposed;receipt["acceptance"]=accepted/proposed
        receipt["records"]=recorded;receipt["final_E"]=copy(state.E)
        receipt["files"]=Dict(rel(p)=>sha(p) for p in (samples,snapshots,checkfile))
        write_toml(receipt_path,receipt)
        println("DONE $(job["run_id"]) seconds=$(receipt["elapsed_seconds"]) legacy_error=$checkerror")
    catch e
        receipt["status"]="failed";receipt["error"]=sprint(showerror,e)
        write_toml(receipt_path,receipt);rethrow()
    end
end

function boundary_main()
    planpath=abspath(arg("plan",""));plan=TOML.parsefile(planpath);boundary_validate(plan)
    job=only(filter(j->j["run_id"]==arg("run-id",""),plan["jobs"]))
    boundary_run(job,plan,planpath)
end
if abspath(PROGRAM_FILE)==@__FILE__;boundary_main();end
