# Buffers for phy_update workflow

mutable struct PhyBuffer_{T<:Number}
	tau::Vector{T}
	ipiv::Vector{LAPACK.BlasInt}

	G::Matrix{T}
	BM::Matrix{T}
	BLs::Array{T,3}
	BRs::Array{T,3}

	# temporaries
    N::Vector{T}
	NN::Matrix{T}
	Nn::Matrix{T}
	nn::Matrix{T}
	nN::Matrix{T}
	zN::Matrix{T}  
end

# Buffers for SCEE workflow
mutable struct G4Buffer_{T<:Number}
    Gt::Matrix{T}
    G0::Matrix{T}
    Gt0::Matrix{T}
    G0t::Matrix{T}
    BLMs::Array{T,3}
    BRMs::Array{T,3}
    BMs::Array{T,3}
    BMinvs::Array{T,3}
end

mutable struct SCEEBuffer_{T<:Number}
    II::Matrix{T}                 # Ns x Ns identity matrix (dense)
    N::Vector{T}                 # Ns
    N_::Vector{T}                # Ns
    zN::Matrix{T}                # 2 x Ns
    nn::Matrix{T}             
    NN::Matrix{T}             
    NN_::Matrix{T}            
    Nn::Matrix{T}             
    nN::Matrix{T}             
    ipiv::Vector{LAPACK.BlasInt}        # length ns
	tau::Vector{T}                # length ns
end

mutable struct AreaBuffer_{T<:Number}
    index::Vector{Int64}          # length nA
    detg::Float64
    gmInv::Matrix{T}          # nA x nA
    NN::Matrix{T}              # nA x nA
    N2::Matrix{T}              # nA x 2
    zN::Matrix{T}              # 2 x nA
    a::Matrix{T}               # nA x 2
    b::Matrix{T}               # 2 x nA
    Tau::Matrix{T}             # 2 x 2
	ipiv::Vector{LAPACK.BlasInt}        # length ns
end


