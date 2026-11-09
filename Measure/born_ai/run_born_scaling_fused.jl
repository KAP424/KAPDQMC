# Production entry point: preserve the original proposal, fuse accepted rank updates.
# The immutable base runner remains the implementation for I/O, entropy and rebuilds.
include("run_born_scaling.jl")
include("fused_original_swap_kernel.jl")

function OriginalSwapScaling.sweep!(state::OriginalSwapScaling.SwapState,
        G::Matrix{Float64},rng::Random.AbstractRNG)
    FusedOriginalSwapKernel.sweep_fused!(state,G,rng)
end

function validate_fused_plan(path)
    plan=validate_plan(path)
    get(plan,"update_kernel","")=="fused_rank_updates" || error("fused entry point requires update_kernel=fused_rank_updates")
    plan["stage"] in ("pilot","pilot_extra","smallcheck") && error("frozen pilot/smallcheck plans must use their original runner")
    any(job["stage"] in ("pilot","pilot_extra","smallcheck") for job in plan["jobs"]) && error("fused plan cannot contain frozen pilot/smallcheck jobs")
    for source in ("run_born_scaling_fused.jl","fused_original_swap_kernel.jl",
                   "run_born_scaling.jl","original_swap_scaling.jl")
        haskey(plan["source_sha256"],source) || error("fused plan must bind source SHA: $source")
        plan["source_sha256"][source]==sha(joinpath(ROOT,source)) || error("fused source hash mismatch: $source")
    end
    method=which(OriginalSwapScaling.sweep!,Tuple{OriginalSwapScaling.SwapState,Matrix{Float64},MersenneTwister})
    normpath(String(method.file))==normpath(@__FILE__) || error("fused wrapper dispatch is not active")
    plan
end

function fused_main()
    flag("prepare-only") && error("prepare a separately authorized frozen production plan; fused entry point cannot prepare pilot plans")
    value=arg("plans",arg("plan","")); isempty(value) && error("provide --plan or --plans")
    for path in abspath.(split(value,',')); validate_fused_plan(path); end
    main()
end

if abspath(PROGRAM_FILE)==@__FILE__; fused_main(); end
