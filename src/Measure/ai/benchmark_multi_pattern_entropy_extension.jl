# One authorized representative size; methods execute sequentially with 8 BLAS threads.
using LinearAlgebra, SHA, Dates, TOML, Printf, Statistics
include(joinpath(@__DIR__,"fermion_experiment.jl"))
include(joinpath(@__DIR__,"real_entropy_extension.jl"))
include(joinpath(@__DIR__,"multi_pattern_entropy_extension.jl"))
import .FermionExperiment as FE
import .RealEntropyExtension as RE
import .MultiPatternEntropyExtension as ME

function evaluate(state,method)
    method==:real_cholesky && return RE.entropy_rectangle_real(state;with_checks=true)
    return ME.entropy_rectangle_shift(state;method,with_checks=true)
end

function main()
    BLAS.set_num_threads(8)
    output=joinpath(@__DIR__,"analysis_multi_pattern_entropy")
    mkpath(output)
    methods=(:real_cholesky,:bunchkaufman,:bunchkaufman_rook,:lu)
    bits=BitVector(c=='1' for c in "100110001011010110")
    warm=FE.measure(FE.evolve(FE.prepare_pattern(FE.BlochModel(18,3),bits),5.0),1.0)
    for method in methods
        evaluate(warm,method)
    end
    L=108;theta=5.0;beta=1.0
    state=FE.measure(FE.evolve(FE.prepare_pattern(FE.BlochModel(L,3),bits),theta),beta)
    results=NamedTuple[]
    csvpath=joinpath(output,"representative_benchmark.csv")
    reference=NaN
    started=time()
    open(csvpath,"w") do io
        println(io,"pattern,L,theta,beta,trial,large_size_warmup,method,S_B,minus_real_cholesky,seconds,allocated_bytes,gc_seconds,max_imaginary,real_part_entropy_bound")
        for trial in 1:3,method in (iseven(trial) ? reverse(methods) : methods)
            GC.gc(true)
            measurement=@timed evaluate(state,method)
            entropy,check=measurement.value
            if isnan(reference)
                @assert method==:real_cholesky
                reference=entropy
            end
            difference=entropy-reference
            @assert abs(difference)<1e-8 (method,difference)
            row=(pattern="random_half_3",L=L,theta=theta,beta=beta,trial=trial,large_size_warmup=trial==1,method=string(method),S_B=entropy,
                 minus_real_cholesky=difference,seconds=measurement.time,allocated_bytes=measurement.bytes,gc_seconds=measurement.gctime,
                 max_imaginary=check.max_imaginary,real_part_entropy_bound=check.real_part_entropy_bound)
            push!(results,row)
            println(io,join(values(row),','));flush(io)
            @printf("L=%d trial=%d %-20s %.3fs difference=%.3g\n",L,trial,method,measurement.time,difference)
        end
    end
    medians=Dict(string(method)=>median(r.seconds for r in results if r.method==string(method)&&r.trial>1) for method in methods)
    files=["benchmark_multi_pattern_entropy_extension.jl","multi_pattern_entropy_extension.jl","fermion_experiment.jl","real_entropy_extension.jl"]
    receipt=Dict("status"=>"PASS","pattern"=>"random_half_3","L"=>L,"theta"=>theta,"beta"=>beta,
        "n"=>4L^2÷9,"trials"=>3,"large_size_warmup_trial"=>1,"warmed_trials"=>[2,3],"blas_threads"=>8,"julia"=>string(VERSION),
        "generated_utc"=>string(now(Dates.UTC)),"elapsed_seconds"=>time()-started,
        "median_seconds"=>medians,"median_relative_to_real"=>Dict(k=>v/medians["real_cholesky"] for (k,v) in medians),
        "max_entropy_difference"=>maximum(abs(r.minus_real_cholesky) for r in results),
        "scope"=>"One representative L108 state, sequential 8-thread BLAS; each timed call includes correlation kernel, covariance construction, factorization and entropy. Common preparation/evolution is excluded; GC runs before each timed call; second trial reverses method order. Trial1 is large-size warmup, medians use trials2/3 only. No large-size sweep.",
        "source_sha256"=>Dict(name=>bytes2hex(sha256(read(joinpath(@__DIR__,name)))) for name in files),
        "data_sha256"=>bytes2hex(sha256(read(csvpath))))
    open(io->TOML.print(io,receipt),joinpath(output,"representative_benchmark.toml"),"w")
    println("PASS timings relative to real: ",receipt["median_relative_to_real"])
end
abspath(PROGRAM_FILE)==abspath(@__FILE__) && main()
