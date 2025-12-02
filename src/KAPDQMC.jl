module KAPDQMC
    # using Base.Filesystem
    using LinearAlgebra,LinearAlgebra.BLAS,LinearAlgebra.LAPACK
    # using DelimitedFiles,Random

    # export LinearAlgebra,DelimitedFiles,Random

    # using 
    # using Statistics

    # Geometry.jl 目前是一个普通脚本文件，并未声明 module Geometry。
    # 因此不能使用 `using Geometry`，而应当 `include` 并从当前模块导出/使用其函数。
    # Geometry 作为子模块，需先 include 再 using 以导入其导出的函数
    include("public/Geometry.jl")
    using .Geometry: nn2idx, xy_i, i_xy, K_Matrix, area_index, nnidx_F

    include("public/Buffer.jl")
    # export PhyBuffer_,G4Buffer_,SCEEBuffer_,AreaBuffer_

   

    include("tU/tUDQMC.jl")
    using .tUDQMC: tU_Hubbard_Para,Initial_s,phy_update
    # include("tV/tVDQMC.jl")
    # include("tUV/tUVDQMC.jl")


    export GroverMatrix,GroverMatrix!

    export Initial_s,phy_update
end


if abspath(PROGRAM_FILE) == @__FILE__
    using .KAPDQMC
    using Random
    path = "test/"

    rng = MersenneTwister(1234)
    
    model=KAPDQMC.tUDQMC.tU_Hubbard_Para(t=1.0, U=4.0, Lattice="HoneyComb120", site=[3,3], Δt=0.1, Θ=1.0, BatchSize=10, Initial="H0")

    s=Initial_s(model,rng)
    phy_update(path,model,s,10,true)
end