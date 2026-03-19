push!(LOAD_PATH, "D:\\JuliaDQMC\\code\\KAPDQMC\\src\\")
using KAPDQMC
using Test
using Random
using LinearAlgebra

@testset "KAPDQMC.jl" begin
    path = "D:\\JuliaDQMC\\code\\KAPDQMC\\test\\tV\\"

    rng = MersenneTwister(time_ns())

    # model = tV_Hubbard_Para(Ht=1.0, Hv1=0.1, Hv2=1.0, Θrelax=0.0, Θquench=1.0, Lattice="HoneyComb120", site=[3, 3], Δt=0.1, BatchSize=5, Initial="H0")

    # s = Initial_s(model, rng)

    # # s=phy_update(path,model,s,10,true)

    # L = model.site[2]
    # indexA = area_index(model.Lattice, model.site, ([1, 1], [div(L, 3), L]))
    # # # HalfHalf
    # indexB = area_index(model.Lattice, model.site, ([1, 1], [div(L, 3), div(2 * L, 3)]))
    # # println(indexB)
    # ss = [copy(s), copy(s)]
    # λ = 0.5
    # Nλ = 2

    # # s = ctrl_SCDOPicr(path, model, π / 2, indexA, indexB, 20, λ, Nλ, s, true)

    # # println(@btime ctrl_SCEEicr($path,$model,$indexA,$indexB,$Sweeps,$λ,$Nλ,$ss,$true) )
    # ss = ctrl_SCEEicr(path, model, indexA, indexB, 2, λ, Nλ, ss, true)

    # ------------------------------------------------------------------------------------------------------------------------------------------------------

    model = tV_Hubbard_Para(Ht=1.0, Hv1=2.0, Hv2=1.30, Θrelax=6.0, Θquench=3.6,
        Lattice="SQUARE90", site=[8, 8], Δt=0.03, BatchSize=5, Initial="V", flux=pi, opt="y")

    # println(length(model.nodes))

    s = Initial_s(model, rng)
    println(size(s))

    # s = phy_update(path, model, s, 2, false)
    # s = phy_update(path, model, s, 80, true)

    L = model.site[2]
    indexA = area_index(model.Lattice, model.site, ([1, 1], [div(L, 2), L]))
    # # HalfHalf
    indexB = area_index(model.Lattice, model.site, ([1, 1], [div(L, 2), div(L, 2)]))
    # println(indexB)
    ss = [copy(s), copy(s)]
    λ = 0.5
    Nλ = 2

    # s = ctrl_SCDOPicr(path, model, π / 2, indexA, indexB, 20, λ, Nλ, s, true)

    # # println(@btime ctrl_SCEEicr($path,$model,$indexA,$indexB,$Sweeps,$λ,$Nλ,$ss,$true) )
    ss = ctrl_SCEEicr(path, model, indexA, indexB, 1, λ, Nλ, ss, false)

end
