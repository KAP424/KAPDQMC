module KAPDQMC
using LinearAlgebra, LinearAlgebra.BLAS, LinearAlgebra.LAPACK

include("public/Geometry.jl")
using .Geometry: K_Matrix, area_index, i_xy, xy_i, nnn2idx, nnnK_Matrix
export area_index, K_Matrix, i_xy, xy_i, nnn2idx, nnnK_Matrix

include("public/Buffer.jl")

include("public/GF.jl")
export Free_G!, GroverMatrix, GroverMatrix!

# Declare unified API to be extended by submodules via multiple dispatch
function phy_update end
function Initial_s end
function ctrl_SCEEicr end
function ctrl_SCDOPicr end

include("tU/tUDQMC.jl")
using .tUDQMC: tU_Hubbard_Para

include("tV/tVDQMC.jl")
using .tVDQMC: tV_Hubbard_Para

include("tUV/tUVDQMC.jl")
using .tUVDQMC: tUV_Hubbard_Para

export tU_Hubbard_Para, tV_Hubbard_Para, tUV_Hubbard_Para
export Initial_s, phy_update, ctrl_SCEEicr, ctrl_SCDOPicr
end


