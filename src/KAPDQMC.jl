module KAPDQMC
using LinearAlgebra, LinearAlgebra.BLAS, LinearAlgebra.LAPACK

include("public/Geometry.jl")
export name_Lattice, nnidx_F, area_index, i_xy, xy_i, nnn2idx, n3n2idx
export nnK_Matrix, nnnK_Matrix, n3nK_Matrix, Initial_Pt!

include("public/Buffer.jl")

include("public/GF.jl")
export Free_G!, GroverMatrix, GroverMatrix!

# Declare unified API to be extended by submodules via multiple dispatch
function phy_update end
function Initial_s end
function ctrl_SCEEicr end
function ctrl_EEicr end
function ctrl_SCDOPicr end

include("tU/tUDQMC.jl")
using .tUDQMC: tU_Hubbard_Para

include("tV/tVDQMC.jl")
using .tVDQMC: tV_Hubbard_Para

include("tUV/tUVDQMC.jl")
using .tUVDQMC: tUV_Hubbard_Para

include("SO3/SO3DQMC.jl")
using .SO3DQMC: SO3_Hubbard_Para, SO3Initial_Pt!, nnK_Matrix4so3, so3area_index
export SO3Initial_Pt!, nnK_Matrix4so3, so3area_index

include("VBS/VBSDQMC.jl")
using .VBSDQMC: VBS_Hubbard_Para

include("M_VBS/M_VBSDQMC.jl")
using .M_VBSDQMC: M_VBS_Hubbard_Para

export tU_Hubbard_Para, tV_Hubbard_Para, tUV_Hubbard_Para, SO3_Hubbard_Para, VBS_Hubbard_Para, M_VBS_Hubbard_Para
export Initial_s, phy_update, ctrl_SCEEicr, ctrl_EEicr, ctrl_SCDOPicr
end


