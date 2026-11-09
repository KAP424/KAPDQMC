# Small-system validation only; never writes production observations or changes FE.
using LinearAlgebra, Statistics, Dates, SHA, TOML
BLAS.set_num_threads(1)
include(joinpath(@__DIR__,"fermion_experiment.jl"))
import .FermionExperiment as FE
include(joinpath(@__DIR__,"beta_patterns_strip_extension.jl"))
import .BetaPatternsStripExtension as FoldedStrip

function main()
    patterns = [(name="one_B_2",m=2,bits="10101001"),
                (name="diagonal_B_3",m=3,bits="011010100110101001"),
                (name="random_half_3",m=3,bits="100110001011010110")]
    output = joinpath(@__DIR__,"analysis_beta_patterns")
    mkpath(output)
    rows = NamedTuple[]
    # Warm both functions, including optional-checks keyword paths.
    warm = FE.measure(FE.evolve(FE.prepare_pattern(FE.BlochModel(18,2),BitVector(c=='1' for c in patterns[1].bits)),4.),.5)
    FE.entropy_strip(warm)
    FoldedStrip.entropy_strip_folded(warm;with_checks=true)
    for p in patterns, L in (18,36)
        model = FE.BlochModel(L,p.m)
        prepared = FE.prepare_pattern(model,BitVector(c=='1' for c in p.bits))
        for theta in (4.,Inf)
            evolved = FE.evolve(prepared,theta)
            for beta in (0.,.5,1.,2.)
                state = FE.measure(evolved,beta)
                for orientation in (:x,:y)
                    original = 0.; folded = 0.; checks = nothing
                    original_seconds = @elapsed original = FE.entropy_strip(state;orientation)
                    folded_seconds = @elapsed (folded,checks) = FoldedStrip.entropy_strip_folded(state;orientation,with_checks=true)
                    error = abs(original-folded)
                    @assert error <= 1e-9
                    @assert checks.total_weight == model.q
                    @assert checks.trim_count == (iseven(model.q) ? 2 : 1)
                    push!(rows,(pattern=p.name,L=L,m=p.m,theta=theta,beta=beta,orientation=String(orientation),
                                original=original,folded=folded,absolute_error=error,
                                original_seconds=original_seconds,folded_seconds=folded_seconds,
                                speed_ratio=original_seconds/folded_seconds,
                                q=checks.q,entropy_evaluations=checks.entropy_evaluations,
                                trim_count=checks.trim_count,max_bloch_conjugacy=checks.max_bloch_conjugacy,
                                max_probe_conjugacy=checks.max_probe_conjugacy,max_trim_imaginary=checks.max_trim_imaginary))
                end
            end
        end
    end
    shifted_max = 0.
    for p in patterns, orientation in (:x,:y)
        state = FE.measure(FE.evolve(FE.prepare_pattern(FE.BlochModel(18,p.m),BitVector(c=='1' for c in p.bits)),4.),1.)
        reference = FE.entropy_strip(state;orientation,width=5,x0=2,copies=2)
        folded = FoldedStrip.entropy_strip_folded(state;orientation,width=5,x0=2,copies=2)
        shifted_max = max(shifted_max,abs(reference-folded))
    end
    @assert shifted_max <= 1e-9
    # Deliberately break real-space covariance by rotating only one momentum block.
    orbitals = deepcopy(warm.orbitals)
    orbitals[2][1,:] .*= cis(.2)
    broken = FE.FermionState(warm.prepared,warm.theta,warm.beta,orbitals)
    rejected = false
    try
        FoldedStrip.entropy_strip_folded(broken)
    catch err
        rejected = occursin("violates real-space conjugacy",sprint(showerror,err))
    end
    @assert rejected
    path = joinpath(output,"strip_validation.csv")
    open(path,"w") do io
        println(io,join(string.(keys(first(rows))),','))
        for row in rows
            println(io,join(values(row),','))
        end
    end
    hashes = Dict(name=>bytes2hex(sha256(read(joinpath(@__DIR__,name)))) for name in
                  ("fermion_experiment.jl","beta_patterns_strip_extension.jl","validate_beta_patterns_strip.jl"))
    receipt = Dict("status"=>"PASS","generated_utc"=>string(now(UTC)),"cases"=>length(rows),
                   "BLAS_threads"=>BLAS.get_num_threads(),"blas_config"=>sprint(show,BLAS.get_config()),
                   "max_absolute_entropy_difference"=>maximum(r.absolute_error for r in rows),
                   "max_bloch_conjugacy"=>maximum(r.max_bloch_conjugacy for r in rows),
                   "max_probe_conjugacy"=>maximum(r.max_probe_conjugacy for r in rows),
                   "max_trim_imaginary"=>maximum(r.max_trim_imaginary for r in rows),
                   "shifted_cut_cases"=>6,"shifted_cut_copies2_max_error"=>shifted_max,
                   "nonreal_covariance_rejected"=>rejected,
                   "median_small_system_speed_ratio"=>median(r.speed_ratio for r in rows),
                   "median_L36_speed_ratio"=>median(r.speed_ratio for r in rows if r.L==36),
                   "code_sha256"=>hashes,"csv_sha256"=>bytes2hex(sha256(read(path))),
                   "scope"=>"Exact opposite-momentum pairing for real covariance; no half-spectrum/sub-lattice symmetry; small-system BLAS1 validation only, not a production benchmark.")
    open(joinpath(output,"strip_validation.toml"),"w") do io
        TOML.print(io,receipt)
    end
    println("PASS ",length(rows)," default strips plus 6 shifted/copies2 controls; max error=",receipt["max_absolute_entropy_difference"],
            "; median speed ratio=",receipt["median_small_system_speed_ratio"])
end

main()
