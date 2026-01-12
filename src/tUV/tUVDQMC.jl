module tUVDQMC
import ..KAPDQMC: phy_update, Initial_s, ctrl_SCEEicr, ctrl_SCDOPicr

using ..KAPDQMC: nn2idx, xy_i, i_xy, nnK_Matrix, area_index, nnidx_F
using ..KAPDQMC: PhyBuffer_, G4Buffer_, SCEEBuffer_, AreaBuffer_, DOPBuffer_
using ..KAPDQMC: inv22!, GroverMatrix, GroverMatrix!

using LinearAlgebra, LinearAlgebra.BLAS, LinearAlgebra.LAPACK
using DelimitedFiles, Random

# 扩展父模块统一 API：导入 `phy_update` 并在本模块中添加方法

include("model.jl")
include("../public/Gupdate.jl")
include("GreenMatrix.jl")
include("phy_update.jl")
include("SCEE.jl")

export tUV_Hubbard_Para

end

