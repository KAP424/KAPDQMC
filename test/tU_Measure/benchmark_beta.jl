using KAPDQMC, Random, LinearAlgebra, SHA

# Run from the repository: julia --project=. test/tU_Measure/benchmark_beta.jl
# A new output directory is required, because the library appends CSV rows.
function main()
    BLAS.set_num_threads(1)
    output = length(ARGS) >= 1 ? abspath(ARGS[1]) : joinpath(@__DIR__, "benchmark_beta_data")
    sweeps = parse(Int, get(ENV, "BENCH_SWEEPS", "10000"))
    warmup = parse(Int, get(ENV, "BENCH_WARMUP", "1000"))
    betas = parse.(Float64, split(get(ENV, "BENCH_BETAS", "0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0"), ','))
    ispath(output) && error("Output already exists: $output. Choose a fresh directory.")
    mkpath(output)
    model = tU_Hubbard_Para(Ht=1.0, Hu1=3.8, Hu2=3.8, Θrelax=5.0, Θquench=0.0,
        Lattice="HoneyComb120", site=[3, 3], Δt=0.1, BatchSize=10, Initial="V")
    Eidx = collect(1:2:model.Ns)
    L = model.site[2]
    indexA = area_index(model.Lattice, model.site, ([1, 1], [div(L, 3), L]))
    indexB = area_index(model.Lattice, model.site, ([1, 1], [div(L, 3), div(2L, 3)]))
    open(joinpath(output, "metadata.txt"), "w") do io
        println(io, "Julia=$VERSION\nsweeps=$sweeps\nwarmup=$warmup\nbetas=$betas\nNlambda=3\nTheta_relax=5.0")
        println(io, "Eidx=$Eidx\nindexA=$indexA\nindexB=$indexB")
        println(io, "Initialization seeds=1235+round(5*beta). Library update RNG uses time_ns; reruns are statistical, not bitwise identical.")
        for relative in ["src/tU/SCEE_M.jl", "test/tU_Measure/SCEEtest_M_tU.jl", "DP_Lib.py"]
            source = normpath(joinpath(@__DIR__, "../..", relative))
            println(io, "$relative sha256=$(bytes2hex(sha256(read(source))))")
        end
    end
    for (i, beta) in enumerate(betas)
        println("START beta=$beta"); flush(stdout)
        rng = MersenneTwister(1235 + round(Int, 5beta))
        s = Initial_s(model, rng)
        ss = [copy(s), copy(s)]
        for k in 0:2
            lambda = k / 3
            ss = ctrl_SCEEicr_M(output, model, Eidx, beta, indexA, indexB, warmup, lambda, 3, ss, false)
            ss = ctrl_SCEEicr_M(output, model, Eidx, beta, indexA, indexB, sweeps, lambda, 3, ss, true)
            flush(stdout)
        end
        println("DONE beta=$beta"); flush(stdout)
    end
end
main()
