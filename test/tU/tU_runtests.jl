push!(LOAD_PATH, "D:\\JuliaDQMC\\code\\KAPDQMC\\src\\")
using KAPDQMC
using Test
using Random
using LinearAlgebra

@testset "KAPDQMC.jl" begin
    path = "test/tU/"

    rng = MersenneTwister(1234)

    model = tU_Hubbard_Para(Ht=1.0, Hu1=0.0, Hu2=3.8, Θrelax=0.0, Θquench=0.1, Lattice="HoneyComb120", site=[3, 3], Δt=0.1, BatchSize=10, Initial="H0")

    s = Initial_s(model, rng)

    s = phy_update(path, model, s, 10, true)


    L = model.site[2]
    indexA = area_index(model.Lattice, model.site, ([1, 1], [div(L, 3), L]))
    # # HalfHalf
    indexB = area_index(model.Lattice, model.site, ([1, 1], [div(L, 3), div(2 * L, 3)]))
    # println(indexB)

    λ = 0.5
    Nλ = 2

    s = ctrl_SCDOPicr(path, model, π / 2, indexA, indexB, 20, λ, Nλ, s, true)


    ss = [copy(s), copy(s)]


    # println(@btime ctrl_SCEEicr($path,$model,$indexA,$indexB,$Sweeps,$λ,$Nλ,$ss,$true) )
    ss = ctrl_SCEEicr(path, model, indexA, indexB, 2, λ, Nλ, ss, true)

    # ------------------------------------------------------------------------------------------------------------------------------------------------------

    model = tU_Hubbard_Para(Ht=1.0, Hu1=4.0, Hu2=4.0, Θrelax=1.0, Θquench=0.0, Lattice="HoneyComb120", site=[3, 3], Δt=0.1, BatchSize=10, Initial="H0")

    s = Initial_s(model, rng)

    s = phy_update(path, model, s, 10, true)


    L = model.site[2]
    indexA = area_index(model.Lattice, model.site, ([1, 1], [div(L, 3), L]))
    # # HalfHalf
    indexB = area_index(model.Lattice, model.site, ([1, 1], [div(L, 3), div(2 * L, 3)]))
    # println(indexB)

    λ = 0.5
    Nλ = 2

    s = ctrl_SCDOPicr(path, model, π / 2, indexA, indexB, 20, λ, Nλ, s, true)


    ss = [copy(s), copy(s)]


    # println(@btime ctrl_SCEEicr($path,$model,$indexA,$indexB,$Sweeps,$λ,$Nλ,$ss,$true) )
    ss = ctrl_SCEEicr(path, model, indexA, indexB, 2, λ, Nλ, ss, true)

end
