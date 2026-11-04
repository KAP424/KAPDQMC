using LinearAlgebra, Random, Printf
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "fermion_experiment.jl"))
using .FermionExperiment

# Independent low-cost reachability screen: no entropy construction.
m = 4
rng = MersenneTwister(20261020)
gamma = BlochModel(4, m)
negative = gamma.vectors[1][:, gamma.energies[1] .< -1e-9]
pool = Dict{String,Float64}()
for _ in 1:4000
    cells = sort(randperm(rng, m^2)[1:m^2÷2])
    occ = BitVector([c in cells for c in 1:m^2 for _ in 1:2])
    sigma = minimum(svdvals(negative[findall(occ), :]))
    sigma > 1e-8 && (pool[join(Int.(occ))] = sigma)
end
candidates = sort(collect(pool), by = x -> (-x.second, x.first))
models = [BlochModel(L, m) for L in (12, 24, 36)]
out = joinpath(@__DIR__, "data", "scan_pure02_m4.csv")
ngood = Ref(0)
tested_gamma = Float64[]
open(out, "w") do io
    println(io, "bits,m,L,gamma_min_sv,excess_energy,min_pivot_sv,reconstruction_error")
    for (rank, (bits, gamma_sv)) in enumerate(candidates)
        rank > 1000 && break
        any(abs(gamma_sv-s) < 1e-8 for s in tested_gamma) && continue
        push!(tested_gamma, gamma_sv)
        occ = BitVector(c == '1' for c in bits)
        results = []
        for model in models
            pre = prepare_pattern(model, occ)
            excess = pre.accessible_energy - model.ground_energy
            push!(results, (model.L, excess, pre.min_pivot_sv, pre.max_reconstruction_error))
            println(io, join((bits,m,model.L,gamma_sv,excess,
                             pre.min_pivot_sv,pre.max_reconstruction_error), ','))
        end
        if all(abs(r[2]) < 1e-8 for r in results)
            ngood[] += 1
            println("PASS ", bits, " gamma=", gamma_sv, " rows=", results)
            flush(stdout)
        end
        ngood[] >= 4 && break
    end
end
println("DONE gamma_unique_pass=",length(pool)," complete_pass=",ngood[]," csv=",out)
