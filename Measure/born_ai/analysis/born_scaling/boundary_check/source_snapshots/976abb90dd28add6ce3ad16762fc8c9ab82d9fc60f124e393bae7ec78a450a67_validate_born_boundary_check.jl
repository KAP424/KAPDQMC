# Independent numerical check for the paired boundary experiment.
# Uses a Gram-normalized projector and eigenvalue entropy, rather than the
# production weighted-orbital QR and purity Cholesky. Never rewrites a receipt.
include("run_born_boundary_check.jl")

function validation_rectangle(L, width, height, tx, ty)
    inside = falses(2L^2)
    for yy in 1:height, xx in 1:width
        x = mod1(xx + tx, L); y = mod1(yy + ty, L)
        cell = x + (y - 1)*L
        inside[2cell-1] = true; inside[2cell] = true
    end
    findall(inside)
end

function validation_projector(model, E, beta)
    P = copy(model.Q)
    P[E,:] .*= exp(-2beta)
    gram = P'P
    factor = cholesky(Symmetric(gram))
    G = P*(factor\P')
    (G + G')/2
end

function validation_s2(G, indices)
    values = eigvals(Symmetric(G[indices,indices]))
    minimum(values) >= -2e-8 && maximum(values) <= 1 + 2e-8 || error("invalid correlation spectrum")
    -sum(log(x*x + (1-x)^2) for x in clamp.(values,0.,1.))
end

function validate_boundary_kernel(; output=joinpath(ROOT,"analysis","born_scaling","boundary_check","kernel_validation.toml"))
    isfile(output) && error("refuse to overwrite validation receipt")
    model = ground_model(6;state=:staggered,mu=1e-8)
    cases = Dict{String,Any}[]
    for (ci,start) in enumerate(("random","uniform_A","uniform_B","02_stripe")), beta in (0.,1.,3.)
        E = initial_set(model.L,start,MersenneTwister(28102026+ci))
        direct_G = validation_projector(model,E,beta)
        group = boundary_entropies!(boundary_workspace(model),model,E,beta)
        full = boundary_entropies!(boundary_workspace(model;full_translations=true),model,E,beta)
        maxerror = 0.; max_complement_error = 0.; group_error = 0.
        for row in full
            w = model.L÷row.m
            regions = (validation_rectangle(model.L,w,model.L,row.tx,row.ty),
                validation_rectangle(model.L,w,model.L-w,row.tx,row.ty),
                validation_rectangle(model.L,model.L,w,row.tx,row.ty))
            s = [validation_s2(direct_G,idx) for idx in regions]
            target = [s...,((row.m-1)*s[1]+s[3])/row.m-s[2]]
            maxerror = max(maxerror,maximum(abs.(target-[row.SA,row.SB,row.SC,row.D])))
            for (i,idx) in enumerate(regions)
                max_complement_error = max(max_complement_error,abs(s[i]-validation_s2(direct_G,setdiff(1:model.N,idx))))
            end
        end
        for row in group
            direct = only(filter(x->x.m==row.m && x.tx==row.tx && x.ty==row.ty,full))
            group_error = max(group_error,maximum(abs.([row.SA-direct.SA,row.SB-direct.SB,row.SC-direct.SC,row.D-direct.D])))
        end
        maxerror < 2e-8 || error("independent entropy check failed")
        max_complement_error < 2e-8 || error("purity complement check failed")
        group_error < 1e-10 || error("subgroup lookup failed")
        max_translation_error = maximum(maximum(r.D for r in full if r.m==m)-minimum(r.D for r in full if r.m==m) for m in (2,3))
        beta != 0. || max_translation_error < 1e-10 || error("beta=0 translation invariance failed")
        push!(cases,Dict("start"=>start,"beta"=>beta,"L"=>model.L,"full_entries"=>length(full),
            "subgroup_entries"=>length(group),"max_independent_entropy_error"=>maxerror,
            "max_complement_error"=>max_complement_error,"max_subgroup_entry_error"=>group_error,
            "max_D_translation_range"=>max_translation_error))
        println("PASS L=6 start=$start beta=$beta entropy_error=$maxerror complement_error=$max_complement_error")
    end
    sources=[@__FILE__,joinpath(ROOT,"run_born_boundary_check.jl"),joinpath(ROOT,"run_born_scaling.jl"),
        joinpath(ROOT,"original_swap_scaling.jl"),joinpath(ROOT,"fused_original_swap_kernel.jl")]
    receipt=Dict("status"=>"PASS","created_at"=>string(now()),"julia_version"=>string(VERSION),
        "method"=>"Independent full translations at L=6; Gram-normalized weighted projector; eigenvalue S2; direct complement identity; production subgroup/full workspace comparison",
        "cases"=>cases,"source_sha256"=>Dict(rel(p)=>sha(p) for p in sources))
    mkpath(dirname(output));write_toml(output,receipt)
    println("WROTE $output")
end

function validate_boundary_samples(planpath; output=nothing)
    planpath=abspath(planpath);plan=TOML.parsefile(planpath)
    boundary_validate(plan)
    stage=plan["stage"]
    output===nothing && (output=joinpath(ROOT,"analysis","born_scaling","boundary_check",stage*"_samples_validation.toml"))
    isfile(output) && error("refuse to overwrite sample validation receipt")
    checks=Dict{String,Any}[]; sources=[@__FILE__,joinpath(ROOT,"run_born_boundary_check.jl")]
    for L in sort(unique(j["L"] for j in plan["jobs"]))
        job=only(filter(j->j["L"]==L && j["chain"]==1,plan["jobs"]))
        receipt_path=joinpath(ROOT,job["receipt_path"])
        receipt=TOML.parsefile(receipt_path)
        receipt["status"]=="complete" || error("chain not complete: $(job["run_id"])")
        receipt["job"]==job || error("job mismatch")
        receipt["plan_sha256"]==sha(planpath) || error("plan hash mismatch")
        for (file,digest) in receipt["files"]
            sha(joinpath(ROOT,file))==digest || error("sample hash mismatch")
        end
        model=load_model(job)
        sample_path=joinpath(ROOT,job["sample_path"])
        snapshot_path=joinpath(ROOT,job["snapshot_path"])
        lines=readlines(sample_path)
        split(lines[1],',')==collect(BOUNDARY_COLUMNS) || error("column mismatch")
        nrecord=job["sweeps"]÷job["measure_every"]
        length(lines)==1+13nrecord || error("incorrect row count")
        filesize(snapshot_path)==sizeof(Int32)*model.n*nrecord || error("incorrect snapshot size")
        targets=[job["analysis_burnin"]÷job["measure_every"]+1,nrecord]
        for record in targets
            E32=Vector{Int32}(undef,model.n)
            open(snapshot_path,"r") do io
                seek(io,sizeof(Int32)*model.n*(record-1));read!(io,E32)
            end
            E=Int.(E32)
            length(unique(E))==model.n && all(i->1<=i<=model.N,E) || error("invalid snapshot set")
            record!=nrecord || E==receipt["final_E"] || error("final snapshot mismatch")
            G=validation_projector(model,E,job["beta"])
            P=copy(model.Q);P[E,:].*=exp(-2job["beta"])
            logweight=2sum(log,diag(cholesky(Symmetric(P'P)).U))-2sum(log,diag(cholesky(Symmetric(model.Q' * model.Q)).U))
            cached=Dict{Tuple,Float64}()
            direct(idx)=get!(cached,Tuple(idx)) do;validation_s2(G,idx);end
            maxerror=0.;max_D_error=0.;maxweight=0.; rows=Dict{String,Any}[]
            for line in lines[(record-1)*13+2:record*13+1]
                cells=split(line,',');row=Dict(k=>v for (k,v) in zip(BOUNDARY_COLUMNS,cells))
                parse(Int,row["record_index"])==record || error("record sequence error")
                parse(Int,row["sweep"])==record*job["measure_every"] || error("sweep sequence error")
                m=parse(Int,row["p_den"]);tx=parse(Int,row["tx"]);ty=parse(Int,row["ty"]);w=L÷m
                indices=(validation_rectangle(L,w,L,tx,ty),validation_rectangle(L,w,L-w,tx,ty),validation_rectangle(L,L,w,tx,ty))
                s=direct.(indices);D=((m-1)*s[1]+s[3])/m-s[2]
                direct_values=[s...,D];stored=parse.(Float64,[row[k] for k in ("SA","SB","SC","D")])
                numerical_error=maximum(abs.(direct_values-stored));maxerror=max(maxerror,numerical_error)
                max_D_error=max(max_D_error,abs(D-stored[4]))
                maxweight=max(maxweight,abs(logweight-parse(Float64,row["log_weight"])))
                push!(rows,Dict("p_den"=>m,"tx"=>tx,"ty"=>ty,"direct_SA"=>s[1],"direct_SB"=>s[2],"direct_SC"=>s[3],"direct_D"=>D,"max_entropy_error"=>numerical_error))
            end
            maxerror<2e-8 || error("snapshot entropy mismatch: $maxerror")
            maxweight<1e-5 || error("snapshot weight mismatch: $maxweight")
            push!(checks,Dict("L"=>L,"chain"=>1,"run_id"=>job["run_id"],"record_index"=>record,
                "sweep"=>record*job["measure_every"],"max_entropy_error"=>maxerror,
                "max_D_error"=>max_D_error,"max_logweight_error"=>maxweight,
                "receipt_sha256"=>sha(receipt_path),"sample_sha256"=>sha(sample_path),
                "snapshot_sha256"=>sha(snapshot_path),"rows"=>rows))
            println("PASS snapshot L=$L record=$record entropy_error=$maxerror logweight_error=$maxweight");flush(stdout)
        end
        GC.gc()
    end
    result=Dict("status"=>"PASS","created_at"=>string(now()),"plan_path"=>rel(planpath),"plan_sha256"=>sha(planpath),
        "method"=>"Chain 1 at every size; first and last retained E snapshots; independent rectangular indexing, Gram-normalized projector and eigenvalue S2 for all 13 translated rows; normalized Gram determinant weight",
        "checks"=>checks,"source_sha256"=>Dict(rel(p)=>sha(p) for p in sources))
    mkpath(dirname(output));write_toml(output,result);println("WROTE $output")
end

if abspath(PROGRAM_FILE)==@__FILE__
    isempty(arg("plan","")) ? validate_boundary_kernel() : validate_boundary_samples(arg("plan",""))
end
