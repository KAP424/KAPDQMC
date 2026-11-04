# Small-size correctness and warmed timing study only; no existing data is changed.
using LinearAlgebra, SHA, Dates, TOML, Printf, Statistics, Random
include(joinpath(@__DIR__,"fermion_experiment.jl"))
include(joinpath(@__DIR__,"real_entropy_extension.jl"))
include(joinpath(@__DIR__,"multi_pattern_entropy_extension.jl"))
import .FermionExperiment as FE
import .RealEntropyExtension as RE
import .MultiPatternEntropyExtension as ME

function main()
    BLAS.set_num_threads(4)
    out=joinpath(@__DIR__,"analysis_multi_pattern_entropy")
    mkpath(out)
    patterns=[("one_B_2",2,"10101001"),
              ("two_B_3",3,"011010101010101001"),
              ("random_half_3",3,"100110001011010110"),
              ("general_matched_a",3,"011010111001011000")]
    methods=(:bunchkaufman,:bunchkaufman_rook,:lu)
    synthetic=0;synthetic_error=0.0
    rng=MersenneTwister(20261019)
    for n in (1,2,7,32),kind in (:zero,:one,:half,:mixed)
        q=Matrix(qr(randn(rng,n,n)).Q)
        values=kind==:zero ? zeros(n) : kind==:one ? ones(n) : kind==:half ? fill(.5,n) : rand(rng,n)
        C=q*Diagonal(values)*q'
        reference=-sum(log.(values.^2 .+ (1 .-values).^2))
        for method in methods,copies in (1,2)
            A=complex.(C)-complex(.5,.5)*I
            entropy=ME.entropy2_shift!(A;method,copies,covariance_validated=true)
            synthetic_error=max(synthetic_error,abs(entropy-copies*reference))
            @assert abs(entropy-copies*reference)<1e-10 (n,kind,method,copies)
            synthetic+=1
        end
    end
    # Warm all methods once before physical-state timings are recorded.
    prewarm=FE.prepare_pattern(FE.BlochModel(18,2),BitVector(c=='1' for c in patterns[1][3]))
    warm=FE.measure(FE.evolve(prewarm,4.0),.3)
    for method in methods
        ME.entropy_rectangle_shift(warm;method)
    end
    FE.entropy_rectangle(warm);RE.entropy_rectangle_real(warm)
    csvpath=joinpath(out,"small_validation.csv")
    timingspath=joinpath(out,"small_warm_timings.csv")
    maximum_error=0.0;count=0;timings=Dict{Tuple{Int,String},Vector{Float64}}()
    started=time()
    open(csvpath,"w") do io
        println(io,"pattern,L,theta,beta,n,method,S2_shift,S2_FE_eigen,S2_real_cholesky,shift_minus_FE,shift_minus_real,seconds_shift,seconds_FE,seconds_real,max_imaginary,real_part_entropy_bound,kernel_hermitian_error,premeasurement_excess")
        for (name,m,bits) in patterns,L in (18,36)
            pre=FE.prepare_pattern(FE.BlochModel(L,m),BitVector(c=='1' for c in bits))
            excess=pre.accessible_energy-pre.model.ground_energy
            for theta in (0.0,4.0,Inf),beta in (0.0,.3,1.0,2.0)
                state=FE.measure(FE.evolve(pre,theta),beta)
                t=time_ns();s_FE=FE.entropy_rectangle(state);t_FE=(time_ns()-t)*1e-9
                t=time_ns();s_real=RE.entropy_rectangle_real(state);t_real=(time_ns()-t)*1e-9
                push!(get!(timings,(L,"FE_eigen"),Float64[]),t_FE)
                push!(get!(timings,(L,"real_cholesky"),Float64[]),t_real)
                for method in methods
                    t=time_ns();s_shift,check=ME.entropy_rectangle_shift(state;method,with_checks=true);elapsed=(time_ns()-t)*1e-9
                    error=max(abs(s_shift-s_FE),abs(s_shift-s_real));maximum_error=max(maximum_error,error)
                    @assert error<1e-8 (name,L,theta,beta,method,error)
                    push!(get!(timings,(L,string(method)),Float64[]),elapsed)
                    println(io,join((name,L,theta,beta,check.n,method,s_shift,s_FE,s_real,s_shift-s_FE,s_shift-s_real,
                        elapsed,t_FE,t_real,check.max_imaginary,check.real_part_entropy_bound,check.kernel_hermitian_error,excess),','))
                end
                count+=1;flush(io)
            end
            @printf("%d/96 cases pattern=%s L=%d max_error=%.3g\n",count,name,L,maximum_error)
        end
    end
    open(timingspath,"w") do io
        println(io,"L,method,cases,median_seconds,min_seconds,max_seconds,median_over_real_cholesky")
        for (L,method) in sort(collect(keys(timings)))
            ts=timings[L,method];baseline=median(timings[L,"real_cholesky"])
            println(io,join((L,method,length(ts),median(ts),minimum(ts),maximum(ts),median(ts)/baseline),','))
        end
    end
    files=["multi_pattern_entropy_extension.jl","validate_multi_pattern_entropy_extension.jl","fermion_experiment.jl","real_entropy_extension.jl"]
    receipt=Dict("status"=>"PASS","physical_cases"=>count,"method_comparisons"=>3count,
        "synthetic_cases"=>synthetic,"synthetic_max_error"=>synthetic_error,"max_entropy_error"=>maximum_error,
        "L"=>[18,36],"theta"=>["0","4","Inf"],"beta"=>[0.0,.3,1.0,2.0],
        "patterns"=>[p[1] for p in patterns],"methods"=>collect(string.(methods)),
        "generated_utc"=>string(now(Dates.UTC)),"elapsed_seconds"=>time()-started,
        "julia"=>string(VERSION),"blas_threads"=>BLAS.get_num_threads(),
        "formula"=>"S2=-n*log(2)-2*logabsdet(C-(1+i)I/2); full real covariance; no half-size assumption",
        "scope"=>"Small-size correctness and warmed end-to-end timings only. No large benchmark; no state/kernel/data replacement. Timing ratios do not predict large-size speedup.",
        "source_sha256"=>Dict(name=>bytes2hex(sha256(read(joinpath(@__DIR__,name)))) for name in files),
        "data_sha256"=>bytes2hex(sha256(read(csvpath))),"timings_sha256"=>bytes2hex(sha256(read(timingspath))))
    open(io->TOML.print(io,receipt),joinpath(out,"small_validation.toml"),"w")
    println("PASS physical=$count synthetic=$synthetic max_error=$maximum_error")
end
abspath(PROGRAM_FILE)==abspath(@__FILE__) && main()
