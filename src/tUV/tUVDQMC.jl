module tUVDQMC
    using ..Geometry: nn2idx, xy_i, i_xy, K_Matrix, area_index, nnidx_F
    using ..KAPDQMC: PhyBuffer_, G4Buffer_, SCEEBuffer_, AreaBuffer_
    using ..KAPDQMC: inv22!, GroverMatrix, GroverMatrix!

    using LinearAlgebra, LinearAlgebra.BLAS, LinearAlgebra.LAPACK
    using DelimitedFiles, Random

    # 扩展父模块统一 API：导入父函数，并在本模块中添加方法
    import ..KAPDQMC: phy_update, Initial_s, ctrl_SCEEicr

    include("model.jl")
    include("../Gupdate.jl")
    include("GreenMatrix.jl")
    include("phy_update.jl")
    include("SCEE.jl")

    export tUV_Hubbard_Para
end

