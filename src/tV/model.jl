# using hopping channel ±1,±2 HS transformation
# add DataType choice: flux=0 Float64, flux≠0 ComplexF64

struct tV_Hubbard_Para_{T<:Number}
    Lattice::String
    Ht::Float64
    Hv1::Float64
    Hv2::Float64
    site::Vector{Int64}
    Θrelax::Float64
    Θquench::Float64
    Ns::Int64
    Nt::Int64
    K::Array{T,2}
    BatchSize::Int64
    Δt::Float64
    α::Vector{Float64}
    γ::Vector{Float64}
    η::Vector{Float64}
    Pt::Array{T,2}
    HalfeK::Array{T,2}
    eK::Array{T,2}
    HalfeKinv::Array{T,2}
    eKinv::Array{T,2}
    nnidx::Matrix{Tuple{Int64,Int64}}
    nodes::Vector{Int64}
    UV::Array{Float64,3}
    samplers_dict::Dict{UInt8,Random.Sampler}
    flux::Float64
end

function tV_Hubbard_Para(; Ht, Hv1, Hv2, Δt, Θrelax, Θquench, Lattice::String, site, BatchSize, Initial::String, flux=0.0, opt="xy")
    T = flux != 0.0 && opt != "y" ? ComplexF64 : Float64

    K = nnK_Matrix(Lattice, site, flux=flux, opt=opt)
    Ns = size(K, 1)

    E, V = LAPACK.syevd!('V', 'L', -Ht .* K[:, :])
    if abs(E[div(Ns, 2)] - E[div(Ns, 2)+1]) > 1e-10
        @warn "Warning: The non-interacting system may be gapped!"
    end
    HalfeK = V * Diagonal(exp.(-Δt .* E ./ 2)) * V'
    eK = V * Diagonal(exp.(-Δt .* E)) * V'
    HalfeKinv = V * Diagonal(exp.(Δt .* E ./ 2)) * V'
    eKinv = V * Diagonal(exp.(Δt .* E)) * V'

    Pt = zeros(T, Ns, div(Ns, 2))
    Initial_Pt!(Lattice, Initial, Pt, K)

    Nt = round(Int, 2 * (Θrelax + Θquench) / Δt)
    if (Θquench > 0) & (abs(Hv1 - Hv2) > 0)
        Hv = LinRange(Hv1, Hv2, round(Int, Θquench / Δt) + 1)[2:end]
        Hv = vcat(fill(Hv1, round(Int, Θrelax / Δt)), collect(Hv), reverse(collect(Hv)), fill(Hv1, round(Int, Θrelax / Δt)))
    else
        @assert (Hv1 == Hv2) & (Θquench < 1e-7) "For Θquench=0, Hv1 must equal Hv2"
        Hv = Hv1 .* ones(Float64, Nt)
    end

    @assert length(Hv) == Nt "Length of Hv profile does not match Nt!"
    @assert norm(reverse(Hv) - Hv) < 1e-10 "HV profile is not symmetric!"

    α = sqrt.(Δt .* Hv ./ 2)
    γ = [1 + sqrt(6) / 3, 1 + sqrt(6) / 3, 1 - sqrt(6) / 3, 1 - sqrt(6) / 3]
    η = [sqrt(2 * (3 - sqrt(6))), -sqrt(2 * (3 - sqrt(6))), sqrt(2 * (3 + sqrt(6))), -sqrt(2 * (3 + sqrt(6)))]

    if div(Nt, 2) % BatchSize == 0
        nodes = collect(0:BatchSize:Nt)
    else
        nodes = vcat(0, reverse(collect(div(Nt, 2)-BatchSize:-BatchSize:1)), collect(div(Nt, 2):BatchSize:Nt), Nt)
    end

    nnidx = nnidx_F(Lattice, site)
    UV = zeros(Float64, Ns, Ns, size(nnidx, 2))
    for j in axes(nnidx, 2)
        for i in axes(nnidx, 1)
            x, y = nnidx[i, j]
            UV[x, x, j] = UV[x, y, j] = UV[y, x, j] = -2^0.5 / 2
            UV[y, y, j] = 2^0.5 / 2
        end
    end

    rng = MersenneTwister(Threads.threadid() + time_ns())
    elements = (1, 2, 3, 4)
    samplers_dict = Dict{UInt8,Random.Sampler}()
    for excluded in elements
        allowed = [i for i in elements if i != excluded]
        samplers_dict[excluded] = Random.Sampler(rng, allowed)
    end

    return tV_Hubbard_Para_{T}(Lattice, Ht, Hv1, Hv2, site, Θrelax, Θquench, Ns,
        Nt, K, BatchSize, Δt, α, γ, η, Pt,
        HalfeK, eK, HalfeKinv, eKinv, nnidx, nodes, UV, samplers_dict, flux)

end

mutable struct UpdateBuffer_{T<:Number}
    acc::Int64
    uv::Matrix{Float64}      # 2 x 2
    tmp22::Matrix{T}   # 2 x 2
    tmp2::Vector{T}    # length 2
    r::Matrix{T}       # 2 x 2
    Δ::Matrix{T}       # 2 x 2
    subidx::Vector{Int64}  # length 2
end

function UpdateBuffer(T)
    uv = [-2^0.5/2 -2^0.5/2; -2^0.5/2 2^0.5/2]
    return UpdateBuffer_(
        0,
        uv,
        Matrix{T}(undef, 2, 2),
        Vector{T}(undef, 2),
        Matrix{T}(undef, 2, 2),
        Matrix{T}(undef, 2, 2),
        Vector{Int64}(undef, 2),
    )
end


# ---------------------------------------------------------------------------------------

function PhyBuffer(T, Ns, NN)
    ns = div(Ns, 2)
    return PhyBuffer_(
        Vector{T}(undef, ns),
        Vector{LAPACK.BlasInt}(undef, ns),
        Matrix{T}(undef, Ns, Ns),
        Matrix{T}(undef, Ns, Ns),
        Array{T}(undef, ns, Ns, NN),
        Array{T}(undef, Ns, ns, NN),
        Vector{T}(undef, Ns),
        Matrix{T}(undef, Ns, Ns),
        Matrix{T}(undef, Ns, ns),
        Matrix{T}(undef, ns, ns),
        Matrix{T}(undef, ns, Ns),
        Matrix{T}(undef, 2, Ns),
    )
end
# ---------------------------------------------------------------------------------------

function SCEEBuffer(T, Ns)
    ns = div(Ns, 2)
    return SCEEBuffer_(
        Matrix{T}(I, Ns, Ns),
        Vector{T}(undef, Ns),
        Vector{T}(undef, Ns),
        Matrix{T}(undef, 2, Ns),
        Matrix{T}(undef, ns, ns),
        Matrix{T}(undef, Ns, Ns),
        Matrix{T}(undef, Ns, Ns),
        Matrix{T}(undef, Ns, ns),
        Matrix{T}(undef, ns, Ns),
        Vector{LAPACK.BlasInt}(undef, ns),
        Vector{T}(undef, ns),
    )
end

function G4Buffer(T, Ns, NN)
    ns = div(Ns, 2)
    return G4Buffer_(
        Matrix{T}(undef, Ns, Ns),
        Matrix{T}(undef, Ns, Ns),
        Matrix{T}(undef, Ns, Ns),
        Matrix{T}(undef, Ns, Ns),
        Array{T,3}(undef, ns, Ns, NN),
        Array{T,3}(undef, Ns, ns, NN),
        Array{T,3}(undef, Ns, Ns, NN),
        Array{T,3}(undef, Ns, Ns, NN),
    )
end

function AreaBuffer(T, index)
    nA = length(index)
    return AreaBuffer_(
        index,
        zero(T),
        Matrix{T}(undef, nA, nA),
        Matrix{T}(undef, nA, nA),
        Matrix{T}(undef, nA, 2),
        Matrix{T}(undef, 2, nA),
        Matrix{T}(undef, nA, 2),
        Matrix{T}(undef, 2, nA),
        Matrix{T}(undef, 2, 2),
        Vector{LAPACK.BlasInt}(undef, nA),
    )
end

# undeveloped 
function DOPBuffer(T, alpha, index)
    nA = length(index)
    return DOPBuffer_(
        alpha,
        index,
        zero(ComplexF64),
        Matrix{T}(undef, nA, nA),
        Matrix{T}(undef, nA, nA),
        Matrix{T}(undef, nA, 1),
        Matrix{T}(undef, 1, nA),
        Matrix{T}(undef, nA, 1),
        Matrix{T}(undef, 1, nA),
        Matrix{T}(undef, 1, 1),
        Vector{LAPACK.BlasInt}(undef, nA),
    )
end

