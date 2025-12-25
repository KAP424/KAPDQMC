# using density channel ±1,±2 HS transformation

struct tUV_Hubbard_Para_
    Lattice::String
    Ht::Float64
    U::Float64
    V::Float64
    a::Float64
    Type::DataType
    site::Vector{Int64}
    Θ::Float64
    Ns::Int64
    Nt::Int64
    K::Array{Float64,2}
    BatchSize::Int64
    Δt::Float64
    γ::Vector{Float64}
    η::Vector{Float64}
    Pt::Array{Float64,2}
    HalfeK::Array{Float64,2}
    eK::Array{Float64,2}
    HalfeKinv::Array{Float64,2}
    eKinv::Array{Float64,2}
    nnidx::Matrix{Tuple{Int64,Int64}}
    nodes::Vector{Int64}
    samplers_dict::Dict{UInt8,Random.Sampler}
end


function tUV_Hubbard_Para(; Ht, Hu, Hv, Lattice::String, site, Δt, Θ, BatchSize, Initial::String)

    K = K_Matrix(Lattice, site)
    Ns = size(K, 1)

    E, V = LAPACK.syevd!('V', 'L', -Ht .* K[:, :])
    HalfeK = V * Diagonal(exp.(-Δt .* E ./ 2)) * V'
    eK = V * Diagonal(exp.(-Δt .* E)) * V'
    HalfeKinv = V * Diagonal(exp.(Δt .* E ./ 2)) * V'
    eKinv = V * Diagonal(exp.(Δt .* E)) * V'

    Pt = zeros(Float64, Ns, div(Ns, 2))
    if Initial == "H0"
        KK = K[:, :]
        # 交错化学势，打开gap，去兼并
        μ = 1e-5
        if occursin("HoneyComb", Lattice)
            KK += μ * Diagonal(repeat([-1, 1], div(Ns, 2)))
        elseif Lattice == "SQUARE"
            for i in 1:Ns
                x, y = i_xy(Lattice, site, i)
                KK[i, i] += μ * (-1)^(x + y)
            end
        end

        # hopping 扰动，避免能级简并
        # KK[KK .!= 0] .+=( rand(size(KK)...) * 1e-3)[KK.!= 0]
        # KK=(KK+KK')./2

        E, V = LAPACK.syevd!('V', 'L', KK[:, :])
        Pt .= V[:, div(Ns, 2)+1:end]
    elseif Initial == "V"
        if occursin("HoneyComb", Lattice)
            for i in 1:div(Ns, 2)
                Pt[i*2-1, i] = 1
            end
        else
            count = 1
            for i in 1:Ns
                x, y = i_xy(Lattice, site, i)
                if (x + y) % 2 == 1
                    Pt[i, count] = 1
                    count += 1
                end
            end
        end
    end
    Pt = HalfeKinv * Pt

    Nt = 2 * cld(Θ, Δt)
    γ = [1 + sqrt(6) / 3, 1 + sqrt(6) / 3, 1 - sqrt(6) / 3, 1 - sqrt(6) / 3]

    z = if Lattice == "SQUARE"
        4.0
    elseif occursin("HoneyComb", Lattice)
        3.0
    end
    @assert Hu^2 > 4 * z^2 * Hv^2 "For stability, require U^2 > 4 z^2 V^2"
    Type = Hu > 0 ? ComplexF64 : Float64
    g = Hu > 0 ? Hu / z / 2 + sqrt(Hu^2 / z^2 / 4 - Hv^2) : Hu / z / 2 - sqrt(Hu^2 / z^2 / 4 - Hv^2)
    a = g == 0 ? 0 : Hv / g
    η = sqrt(abs(Δt * g)) .* [sqrt(3 - sqrt(6)), -sqrt(3 - sqrt(6)), sqrt(3 + sqrt(6)), -sqrt(3 + sqrt(6))]

    tmp = nnidx_F(Lattice, site)
    nnidx = Matrix{Tuple{Int64,Int64}}(undef, Ns, size(tmp, 2))
    for i in axes(tmp, 1)
        for j in axes(tmp, 2)
            nnidx[2*i-1, j] = tmp[i, j]
            nnidx[2*i, j] = reverse(tmp[i, j])
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

    return tUV_Hubbard_Para_(Lattice, Ht, Hu, Hv, a, Type, site, Θ, Ns, Nt, K, BatchSize, Δt, γ, η, Pt, HalfeK, eK, HalfeKinv, eKinv, nnidx, nodes, samplers_dict)

end

mutable struct UpdateBuffer_{T<:Number}
    a::Float64
    tmp22::Matrix{T}          # 2×2
    tmp2::Vector{T}           # length 2
    r::Matrix{T}              # 2×2
    Δ::Diagonal{T,Vector{T}}  # diag length 2
    subidx::Vector{Int}       # length 2
end

function UpdateBuffer(::Type{T}) where {T<:Number}
    return UpdateBuffer_{T}(
        0.0,
        Matrix{T}(undef, 2, 2),
        Vector{T}(undef, 2),
        Matrix{T}(undef, 2, 2),
        Diagonal(zeros(T, 2)),   # 或 Diagonal(Vector{T}(undef,2)) 若确定后面会全部覆盖
        Vector{Int}(undef, 2)
    )
end


# ---------------------------------------------------------------------------------------

function PhyBuffer(::Type{T}, Ns, NN) where {T<:Number}
    ns = div(Ns, 2)
    return PhyBuffer_(
        Vector{T}(undef, ns),
        Vector{LAPACK.BlasInt}(undef, ns), Matrix{T}(undef, Ns, Ns),
        Matrix{T}(undef, Ns, Ns),
        Array{T}(undef, ns, Ns, NN),
        Array{T}(undef, Ns, ns, NN), Vector{T}(undef, Ns),
        Matrix{T}(undef, Ns, Ns),
        Matrix{T}(undef, Ns, ns),
        Matrix{T}(undef, ns, ns),
        Matrix{T}(undef, ns, Ns),
        Matrix{T}(undef, 2, Ns),
    )
end
# ---------------------------------------------------------------------------------------


function SCEEBuffer(::Type{T}, Ns) where {T<:Number}
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

function G4Buffer(::Type{T}, Ns, NN) where {T<:Number}
    ns = div(Ns, 2)
    return G4Buffer_(
        Matrix{T}(undef, Ns, Ns),
        Matrix{T}(undef, Ns, Ns),
        Matrix{T}(undef, Ns, Ns),
        Matrix{T}(undef, Ns, Ns), Array{T,3}(undef, ns, Ns, NN),
        Array{T,3}(undef, Ns, ns, NN),
        Array{T,3}(undef, Ns, Ns, NN),
        Array{T,3}(undef, Ns, Ns, NN),
    )
end

function AreaBuffer(::Type{T}, index) where {T<:Number}
    nA = length(index)
    return AreaBuffer_(
        index,
        0.0,
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

