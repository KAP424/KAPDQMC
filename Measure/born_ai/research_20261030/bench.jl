# Where does the per-sample time go, and does BLAS threading help?
#
# Usage: julia --project=@v1.11 bench.jl [nblas_threads]

using LinearAlgebra, Random, Printf

nthr = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 1
BLAS.set_num_threads(nthr)

include(joinpath(@__DIR__, "..", "research_20261029", "StratifiedBorn.jl"))
using .StratifiedBorn
include(joinpath(@__DIR__, "..", "original_swap_scaling.jl"))
using .OriginalSwapScaling

function bench(L, beta, nrep)
    t00 = time()
    m = ground_model(L; state=:staggered, mu=1e-8)
    work = EntropyWorkspace(m)
    s = StratifiedSampler(m.Q, beta)
    tsetup = time() - t00

    rng = MersenneTwister(1234)
    draws = [sample_stratum(s, 0, m.n; rng) for _ in 1:nrep]

    t0 = time(); for _ in 1:nrep; sample_stratum(s, 0, m.n; rng); end; t1 = time()
    for d in draws; entropy!(work, m, d.E, beta); end      # warm up
    t2 = time(); for d in draws; entropy!(work, m, d.E, beta); end; t3 = time()

    ts = 1000 * (t1 - t0) / nrep
    te = 1000 * (t3 - t2) / nrep
    @printf("L=%2d  M=%4d n=%4d  setup %5.1f s | sample %7.1f ms | entropy %7.1f ms | total %7.1f ms  (BLAS %d)\n",
            L, size(m.Q, 1), m.n, tsetup, ts, te, ts + te, BLAS.get_num_threads())
    flush(stdout)
end

for (L, beta, nr) in ((12, 3.6945, 10), (18, 4.0999, 5), (24, 4.3876, 3), (30, 4.6108, 2))
    bench(L, beta, nr)
end
