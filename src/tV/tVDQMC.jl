module tVDQMC
    using LinearAlgebra,LinearAlgebra.BLAS,LinearAlgebra.LAPACK
    using Random, DelimitedFiles
    include("model.jl")
    # 使用父模块中已加载的公共几何算法符号
    using ..Geometry: nn2idx, xy_i, i_xy, K_Matrix, area_index, nnidx_F
    # 导入公共 Buffer 与 GF 工具（由父模块提供）
    using ..KAPDQMC: PhyBuffer_, G4Buffer_, SCEEBuffer_, AreaBuffer_
    using ..KAPDQMC: Free_G!, Gupdate!, G4update!, GMupdate!, get_abTau1!, get_abTau2!, get_G!, inv22!, GroverMatrix, GroverMatrix!
    include("GreenMatrix.jl")
    include("phy_update.jl")
    include("SCEE.jl")

    export tV_Hubbard_Para
end