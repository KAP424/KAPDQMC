push!(LOAD_PATH, "D:\\JuliaDQMC\\code\\KAPDQMC\\src\\")
using KAPDQMC
using Test
using Random
using LinearAlgebra

@testset "KAPDQMC.jl" begin
    path = "test/tV/"

    rng = MersenneTwister(1234)

    model = tV_Hubbard_Para(Ht=1.0, Hv1=0.1, Hv2=1.0, Θrelax=0.0, Θquench=1.0, Lattice="HoneyComb120", site=[3, 3], Δt=0.1, BatchSize=5, Initial="H0")

    s = Initial_s(model, rng)

    # s=phy_update(path,model,s,10,true)

    L = model.site[2]
    indexA = area_index(model.Lattice, model.site, ([1, 1], [div(L, 3), L]))
    # # HalfHalf
    indexB = area_index(model.Lattice, model.site, ([1, 1], [div(L, 3), div(2 * L, 3)]))
    # println(indexB)
    ss = [copy(s), copy(s)]
    λ = 0.5
    Nλ = 2

    # s = ctrl_SCDOPicr(path, model, π / 2, indexA, indexB, 20, λ, Nλ, s, true)

    # println(@btime ctrl_SCEEicr($path,$model,$indexA,$indexB,$Sweeps,$λ,$Nλ,$ss,$true) )
    ss = ctrl_SCEEicr(path, model, indexA, indexB, 2, λ, Nλ, ss, true)

    # ------------------------------------------------------------------------------------------------------------------------------------------------------

    model = tV_Hubbard_Para(Ht=1.0, Hv1=1.0, Hv2=1.0, Θrelax=1.0, Θquench=0.0, Lattice="HoneyComb120", site=[3, 3], Δt=0.05, BatchSize=5, Initial="H0")

    s = Initial_s(model, rng)

    # s=phy_update(path,model,s,10,true)

    L = model.site[2]
    indexA = area_index(model.Lattice, model.site, ([1, 1], [div(L, 3), L]))
    # # HalfHalf
    indexB = area_index(model.Lattice, model.site, ([1, 1], [div(L, 3), div(2 * L, 3)]))
    # println(indexB)
    ss = [copy(s), copy(s)]
    λ = 0.5
    Nλ = 2

    # s = ctrl_SCDOPicr(path, model, π / 2, indexA, indexB, 20, λ, Nλ, s, true)

    # println(@btime ctrl_SCEEicr($path,$model,$indexA,$indexB,$Sweeps,$λ,$Nλ,$ss,$true) )
    ss = ctrl_SCEEicr(path, model, indexA, indexB, 2, λ, Nλ, ss, true)

end
