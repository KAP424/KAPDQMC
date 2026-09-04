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

    model = tV_Hubbard_Para(Ht=1.0, Hv1=2.5, Hv2=1.35, Θrelax=5.1, Θquench=3.6,
        Lattice="HoneyComb120", site=[9, 9], Δt=0.03, BatchSize=5, Initial="V", flux=0, opt="y")

    # G0 = I(model.Ns) - model.Pt * inv(model.Pt' * model.Pt) * model.Pt'
    # # println(diag(G0))
    # Phy = KAPDQMC.tVDQMC.PhyBuffer(Float64, model.Ns, length(model.nodes))
    # Phy.G .= G0
    # tmp = KAPDQMC.tVDQMC.phy_measure(model, Phy, div(model.Nt, 2), Initial_s(model, rng))
    # println(tmp)
    # println(sum(tmp[3] .* [-1, -1, 1, 1]), "  ", sum(tmp[4] .* [-1, -1, 1, 1]))
    # println(1 - sum(tmp[4] .* [-1, -1, 1, 1]) / sum(tmp[3] .* [-1, -1, 1, 1]))


    # λ = 0.5
    # Nλ = 2

    # s = ctrl_SCDOPicr(path, model, π / 2, indexA, indexB, 20, λ, Nλ, s, true)

    # println(@btime ctrl_SCEEicr($path,$model,$indexA,$indexB,$Sweeps,$λ,$Nλ,$ss,$true) )
    # ss = ctrl_SCEEicr(path, model, indexA, indexB, 2, λ, Nλ, ss, true)


    # # println(length(model.nodes))

    s = Initial_s(model, rng)
    # # println(size(s))

    # # # s = phy_update(path, model, s, 2, false)
    # s = phy_update(path, model, s, 2, true)


    L = model.site[2]
    indexA = area_index(model.Lattice, model.site, ([1, 1], [div(L, 3), L]))
    # # HalfHalf
    indexB = area_index(model.Lattice, model.site, ([1, 1], [div(L, 3), div(2 * L, 3)]))
    # println(indexB)
    ss = [copy(s), copy(s)]
    λ = 0.5
    Nλ = 2
    ss = ctrl_SCEEicr(path, model, indexA, indexB, 3, λ, Nλ, ss, true)

    # s = ctrl_SCDOPicr(path, model, π / 2, indexA, indexB, 20, λ, Nλ, s, true)

    # # println(@btime ctrl_SCEEicr($path,$model,$indexA,$indexB,$Sweeps,$λ,$Nλ,$ss,$true) )

end
