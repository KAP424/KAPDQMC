push!(LOAD_PATH, "D:\\JuliaDQMC\\code\\KAPDQMC\\src\\")
using KAPDQMC
using Test
using Random
using LinearAlgebra

@testset "KAPDQMC.jl" begin
    path = "test/tU_Measure/"

    rng = MersenneTwister(1234)

    model = tU_Hubbard_Para(Ht=1.0, Hu1=3.8, Hu2=3.8, Θrelax=2., Θquench=0., Lattice="HoneyComb120", site=[3, 3], Δt=0.1, BatchSize=10, Initial="V")
    s = Initial_s(model, rng)
    ss = [copy(s), copy(s)]

    println(model.nodes)

    beta = 0.
    Eidx = collect(1:2:model.Ns)

    L = model.site[2]
    indexA = area_index(model.Lattice, model.site, ([1, 1], [div(L, 3), L]))
    indexB = area_index(model.Lattice, model.site, ([1, 1], [div(L, 3), div(2 * L, 3)]))

    λ = 0.
    Nλ = 1
    ss = ctrl_SCEEicr_M(path, model, Eidx, beta, indexA, indexB, 500, λ, Nλ, ss, true)
end