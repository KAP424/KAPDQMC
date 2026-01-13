# using SU(2) ±1,±2 HS transformation
mutable struct UpdateBuffer_
    r::Matrix{ComplexF64}
    subidx::Vector{Int64}
end

function UpdateBuffer()
    return UpdateBuffer_(
        Matrix{ComplexF64}(undef, 1, 1),
        [0],
    )
end

struct tU_Hubbard_Para_
    Lattice::String
    Ht::Float64
    Hu1::Float64
    Hu2::Float64
    site::Vector{Int64}
    Θrelax::Float64
    Θquench::Float64
    Ns::Int64
    Nt::Int64
    K::Array{ComplexF64,2}
    BatchSize::Int64
    Δt::Float64
    α::Vector{Float64}
    γ::Vector{Float64}
    η::Vector{Float64}
    Pt::Array{ComplexF64,2}
    HalfeK::Array{ComplexF64,2}
    eK::Array{ComplexF64,2}
    HalfeKinv::Array{ComplexF64,2}
    eKinv::Array{ComplexF64,2}
    nodes::Vector{Int64}
    samplers_dict::Dict{UInt8,Random.Sampler}
end

function tU_Hubbard_Para(; Ht, Hu1, Hu2, Δt, Θrelax, Θquench, Lattice::String, site, BatchSize, Initial::String, flux=0.0)
    Nt = round(Int, 2 * (Θrelax + Θquench) / Δt)
    if (Θquench > 0.0) & (abs(Hu1 - Hu2) > 0)
        # ΔU = (Hu1 - Hu2) / Θquench * Δt
        # Hu22 = vcat(fill(Hu1, round(Int, Θrelax / Δt)), reverse(collect(Hu2:ΔU:Hu1-ΔU/2)), collect(Hu2:ΔU:Hu1-ΔU/2), fill(Hu1, round(Int, Θrelax / Δt)))
        Hu = LinRange(Hu1, Hu2, round(Int, Θquench / Δt) + 1)[2:end]
        Hu = vcat(fill(Hu1, round(Int, Θrelax / Δt)), collect(Hu), reverse(collect(Hu)), fill(Hu1, round(Int, Θrelax / Δt)))
    else
        @assert (Hu1 == Hu2) & (Θquench < 1e-7) "For Θquench=0, Hu1 must equal Hu2"
        Hu = Hu1 .* ones(Float64, Nt)
    end

    @assert norm(reverse(Hu) - Hu) < 1e-10 "HU profile is not symmetric!"
    @assert length(Hu) == Nt "Length of Hu profile does not match Nt!"

    α = sqrt.(Δt .* Hu ./ 2)
    γ = [1 + sqrt(6) / 3, 1 + sqrt(6) / 3, 1 - sqrt(6) / 3, 1 - sqrt(6) / 3]
    η = [sqrt(2 * (3 - sqrt(6))), -sqrt(2 * (3 - sqrt(6))), sqrt(2 * (3 + sqrt(6))), -sqrt(2 * (3 + sqrt(6)))]

    K = nnK_Matrix(Lattice, site, flux=flux)
    Ns = size(K, 1)

    E, V = LAPACK.syevd!('V', 'L', Ht * K[:, :])
    HalfeK = V * Diagonal(exp.(-Δt .* E ./ 2)) * V'
    eK = V * Diagonal(exp.(-Δt .* E)) * V'
    HalfeKinv = V * Diagonal(exp.(Δt .* E ./ 2)) * V'
    eKinv = V * Diagonal(exp.(Δt .* E)) * V'

    Pt = zeros(ComplexF64, Ns, div(Ns, 2))  # 预分配 Pt
    if Initial == "H0"
        KK = copy(K)
        μ = 1e-5
        KK .+= μ * diagm(repeat([-1, 1], div(Ns, 2)))
        E, V = LAPACK.syevd!('V', 'L', KK)
        Pt .= V[:, 1:div(Ns, 2)]
    elseif Initial == "V"
        for i in 1:div(Ns, 2)
            Pt[i*2, i] = 1
        end
    end

    if div(Nt, 2) % BatchSize == 0
        nodes = collect(0:BatchSize:Nt)
    else
        nodes = vcat(0, reverse(collect(div(Nt, 2)-BatchSize:-BatchSize:1)), collect(div(Nt, 2):BatchSize:Nt), Nt)
    end

    rng = MersenneTwister(Threads.threadid() + time_ns())
    elements = (1, 2, 3, 4)
    samplers_dict = Dict{UInt8,Random.Sampler}()
    for excluded in elements
        allowed = [i for i in elements if i != excluded]
        samplers_dict[excluded] = Random.Sampler(rng, allowed)
    end

    return tU_Hubbard_Para_(Lattice, Ht, Hu1, Hu2, site, Θrelax, Θquench, Ns, Nt, K, BatchSize, Δt, α, γ, η, Pt, HalfeK, eK, HalfeKinv, eKinv, nodes, samplers_dict)
end

function PhyBuffer(Ns, NN)
    ns = div(Ns, 2)
    return PhyBuffer_(
        Vector{ComplexF64}(undef, ns),
        Vector{LAPACK.BlasInt}(undef, ns), Matrix{ComplexF64}(undef, Ns, Ns),
        Matrix{ComplexF64}(undef, Ns, Ns),
        Array{ComplexF64}(undef, ns, Ns, NN),
        Array{ComplexF64}(undef, Ns, ns, NN), Vector{ComplexF64}(undef, Ns),
        Matrix{ComplexF64}(undef, Ns, Ns),
        Matrix{ComplexF64}(undef, Ns, ns),
        Matrix{ComplexF64}(undef, ns, ns),
        Matrix{ComplexF64}(undef, ns, Ns),
        Matrix{ComplexF64}(undef, 1, Ns),
    )
end
# ---------------------------------------------------------------------------------------
# Buffers for SCEE workflow

function SCEEBuffer(Ns)
    ns = div(Ns, 2)
    return SCEEBuffer_(
        Matrix{ComplexF64}(I, Ns, Ns),
        Vector{ComplexF64}(undef, Ns),
        Vector{ComplexF64}(undef, Ns),
        Matrix{ComplexF64}(undef, 1, Ns),
        Matrix{ComplexF64}(undef, ns, ns),
        Matrix{ComplexF64}(undef, Ns, Ns),
        Matrix{ComplexF64}(undef, Ns, Ns),
        Matrix{ComplexF64}(undef, Ns, ns),
        Matrix{ComplexF64}(undef, ns, Ns),
        Vector{LAPACK.BlasInt}(undef, ns),
        Vector{ComplexF64}(undef, ns),
    )
end

function G4Buffer(Ns, NN)
    ns = div(Ns, 2)
    return G4Buffer_(
        Matrix{ComplexF64}(undef, Ns, Ns),
        Matrix{ComplexF64}(undef, Ns, Ns),
        Matrix{ComplexF64}(undef, Ns, Ns),
        Matrix{ComplexF64}(undef, Ns, Ns), Array{ComplexF64,3}(undef, ns, Ns, NN),
        Array{ComplexF64,3}(undef, Ns, ns, NN),
        Array{ComplexF64,3}(undef, Ns, Ns, NN),
        Array{ComplexF64,3}(undef, Ns, Ns, NN),
    )
end

function AreaBuffer(index)
    nA = length(index)
    return AreaBuffer_(
        index,
        0.0,
        Matrix{ComplexF64}(undef, nA, nA),
        Matrix{ComplexF64}(undef, nA, nA),
        Matrix{ComplexF64}(undef, nA, 1),
        Matrix{ComplexF64}(undef, 1, nA),
        Matrix{ComplexF64}(undef, nA, 1),
        Matrix{ComplexF64}(undef, 1, nA),
        Matrix{ComplexF64}(undef, 1, 1),
        Vector{LAPACK.BlasInt}(undef, nA),
    )
end

function DOPBuffer(alpha, index)
    nA = length(index)
    return DOPBuffer_(
        alpha,
        index,
        0.0,
        Matrix{ComplexF64}(undef, nA, nA),
        Matrix{ComplexF64}(undef, nA, nA),
        Matrix{ComplexF64}(undef, nA, 1),
        Matrix{ComplexF64}(undef, 1, nA),
        Matrix{ComplexF64}(undef, nA, 1),
        Matrix{ComplexF64}(undef, 1, nA),
        Matrix{ComplexF64}(undef, 1, 1),
        Vector{LAPACK.BlasInt}(undef, nA),
    )
end
