module tUDQMC
    using ..Geometry: nn2idx, xy_i, i_xy, K_Matrix, area_index, nnidx_F
    # 子模块有独立作用域，父模块的 using 不会自动生效。
    # 最简单稳妥的做法：在本模块再次 using 需要的包（不会重复加载，只是引入符号）。
    using LinearAlgebra, LinearAlgebra.BLAS, LinearAlgebra.LAPACK
    using DelimitedFiles, Random

    # 从父模块导入公共 Buffer 类型与 GF 工具函数，避免重复 include
    using ..KAPDQMC: PhyBuffer_, G4Buffer_, SCEEBuffer_, AreaBuffer_

    include("model.jl")
    include("../public/GF.jl")
    include("GreenMatrix.jl")
    include("phy_update.jl")
    include("SCEE.jl")

    export tU_Hubbard_Para
    export Initial_s, phy_update
end


if abspath(PROGRAM_FILE) == @__FILE__

    using .tUDQMC
    using Random
    path = "test/"

    rng = MersenneTwister(1234)
    model=tU_Hubbard_Para(t=1.0, U=4.0, Lattice="HoneyComb120", site=[3,3], Δt=0.1, Θ=1.0, BatchSize=10, Initial="H0")

    s=Initial_s(model,rng)
    phy_update(path,model,s,10,true)
end