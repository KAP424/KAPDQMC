# Direct same-E comparison. The original generic method is still reachable.
include("run_born_boundary_check_fast.jl")

function validate_fast_boundary()
    output=joinpath(ROOT,"analysis","born_scaling","boundary_check","fast_kernel_validation.toml")
    isfile(output) && error("refuse to overwrite fast-kernel validation")
    planpath=joinpath(ROOT,"data","born_scaling","boundary_check","pilot","plan.toml")
    plan=TOML.parsefile(planpath);boundary_validate(plan)
    records=Dict{String,Any}[]
    old_method=which(boundary_entropies!,Tuple{Any,Any,Any,Any})
    normpath(String(old_method.file))==normpath(joinpath(ROOT,"run_born_boundary_check.jl")) || error("original method was replaced")
    for L in (6,18,36)
        job=only(filter(j->j["L"]==L && j["chain"]==1,plan["jobs"]))
        model=load_model(job);E=initial_set(L,job["start"],MersenneTwister(job["seed"]))
        work=boundary_workspace(model)
        old=nothing;fast=nothing
        old_seconds=@elapsed old=invoke(boundary_entropies!,Tuple{Any,Any,Any,Any},work,model,E,job["beta"])
        fast_seconds=@elapsed fast=boundary_entropies!(work,model,E,job["beta"])
        length(old)==length(fast)==13 || error("incorrect row count")
        maxerror=0.
        for (a,b) in zip(old,fast)
            (a.m,a.tx,a.ty,a.tid,a.ids)==(b.m,b.tx,b.ty,b.tid,b.ids) || error("geometry changed")
            maxerror=max(maxerror,maximum(abs.([a.SA-b.SA,a.SB-b.SB,a.SC-b.SC,a.D-b.D])))
        end
        maxerror<1e-9 || error("optimized entropy mismatch: $maxerror")
        # L18 repeated after compilation is a modest timing control. Timings are
        # explicitly concurrent with the pilot and are not isolated benchmarks.
        old_repeat=Float64[];fast_repeat=Float64[]
        if L==18
            for _ in 1:2
                push!(old_repeat,@elapsed invoke(boundary_entropies!,Tuple{Any,Any,Any,Any},work,model,E,job["beta"]))
                push!(fast_repeat,@elapsed boundary_entropies!(work,model,E,job["beta"]))
            end
        end
        push!(records,Dict("L"=>L,"beta"=>job["beta"],"seed"=>job["seed"],"entries"=>length(old),
            "max_entropy_difference"=>maxerror,"old_first_seconds"=>old_seconds,"fast_first_seconds"=>fast_seconds,
            "old_repeat_seconds"=>old_repeat,"fast_repeat_seconds"=>fast_repeat))
        println("PASS fast L=$L max_error=$maxerror old_seconds=$old_seconds fast_seconds=$fast_seconds");flush(stdout)
        GC.gc()
    end
    sources=[@__FILE__,joinpath(ROOT,"run_born_boundary_check_fast.jl"),joinpath(ROOT,"run_born_boundary_check.jl"),
        joinpath(ROOT,"original_swap_scaling.jl")]
    receipt=Dict("status"=>"PASS","created_at"=>string(now()),"julia_version"=>string(VERSION),
        "method"=>"Same initial E and Q at L=6,18,36, beta=3, all 13 paired geometry/translation rows; generic original versus specialized full-correlation method",
        "timing_scope"=>"Run concurrently with original pilot; first call includes compilation; only L18 repeats exclude compilation. Not an isolated wall-time benchmark.",
        "pilot_plan_sha256"=>sha(planpath),"checks"=>records,"source_sha256"=>Dict(rel(p)=>sha(p) for p in sources))
    mkpath(dirname(output));write_toml(output,receipt)
    for p in sources
        snapshot=joinpath(dirname(output),"source_snapshots",sha(p)*"_"*basename(p))
        isfile(snapshot) || cp(p,snapshot)
    end
    println("WROTE $output")
end

if abspath(PROGRAM_FILE)==@__FILE__;validate_fast_boundary();end
