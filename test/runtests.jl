push!(LOAD_PATH,"C:/Users/admin/Desktop/JuliaDQMC/code/KAPDQMC/src/")
using KAPDQMC
using Test
using Random
using LinearAlgebra

@testset "KAPDQMC.jl" begin
    path = "test/"
    rng = MersenneTwister(1234)
    model=tU_Hubbard_Para(t=1.0, U=4.0, Lattice="HoneyComb120", site=[3,3], Δt=0.1, Θ=1.0, BatchSize=10, Initial="H0")
    
    s=Initial_s(model,rng)
    phy_update(path,model,s,10,true)

    # Write your tests here.
end
