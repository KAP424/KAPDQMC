# using hopping channel ±1,±2 HS transformation
# add DataType choice: flux=0 Float64, flux≠0 ComplexF64

struct M_VBS_Hubbard_Para_
    SUN::Int64
    Lattice::String
    Ht::Float64
    HJ1::Float64
    HJ2::Float64
    site::Vector{Int64}
    Θrelax::Float64
    Θquench::Float64
    Ns::Int64
    Nt::Int64
    K::Array{ComplexF64,2}
    BatchSize::Int64
    Δt::Float64
    # α::Vector{Float64}
    # η::Vector{Float64}
    exp_αη_pos::Matrix{Float64}  # 大小: length(α) × 4
    exp_αη_neg::Matrix{Float64}  # 大小: length(α) × 4
    αη::Matrix{Float64}  # 大小: length(α) × 4
    γ::Vector{Float64}
    Pt::Array{ComplexF64,2}
    HalfeK::Array{ComplexF64,2}
    eK::Array{ComplexF64,2}
    HalfeKinv::Array{ComplexF64,2}
    eKinv::Array{ComplexF64,2}
    nnidx::Matrix{Tuple{Int64,Int64}}
    nodes::Vector{Int64}
    UV::Array{ComplexF64,3}
    samplers_dict::Dict{UInt8,Random.Sampler}
    flux::Float64
end

function M_VBS_Hubbard_Para(; SUN, Ht, HJ1, HJ2, Δt, Θrelax, Θquench, Lattice::String, site, BatchSize, Initial::String, flux=0.0, opt="xy")
    Ns = prod(site) * 2
    nnidx = nnidx_F(Lattice, site)
    # println(nnidx)
    realK = zeros(Float64, Ns, Ns)
    K = zeros(ComplexF64, Ns, Ns)
    for (x, y) in nnidx
        realK[x, y] = 1 / 2
        realK[y, x] = -1 / 2
        K[x, y] = 1im / 2
        K[y, x] = -1im / 2
    end

    E, V = LAPACK.syevd!('V', 'L', -Ht .* K[:, :])

    if abs(E[div(Ns, 2)] - E[div(Ns, 2)+1]) > 1e-10
        @warn "Warning: The non-interacting system may be gapped!"
    end

    # K 矩阵纯虚
    HalfeK = V * Diagonal(exp.(-Δt .* E ./ 2)) * V'
    eK = V * Diagonal(exp.(-Δt .* E)) * V'
    HalfeKinv = V * Diagonal(exp.(Δt .* E ./ 2)) * V'
    eKinv = V * Diagonal(exp.(Δt .* E)) * V'

    Pt = zeros(ComplexF64, Ns, div(Ns, 2))
    if Initial != "M_VBS" && Initial != "H0"
        error("Majorana channel Only support Initial M_VBS and H0")
    end

    Initial_Pt!(Lattice, Initial, Pt, K)

    Nt = round(Int, 2 * (Θrelax + Θquench) / Δt)
    if (Θquench > 0) & (abs(HJ1 - HJ2) > 0)
        HJ = LinRange(HJ1, HJ2, round(Int, Θquench / Δt) + 1)[2:end]
        HJ = vcat(fill(HJ1, round(Int, Θrelax / Δt)), collect(HJ), reverse(collect(HJ)), fill(HJ1, round(Int, Θrelax / Δt)))
    else
        @assert (HJ1 == HJ2) & (Θquench < 1e-7) "For Θquench=0, HJ1 must equal HJ2"
        HJ = HJ1 .* ones(Float64, Nt)
    end

    @assert length(HJ) == Nt "Length of HJ profile does not match Nt!"
    @assert norm(reverse(HJ) - HJ) < 1e-10 "HV profile is not symmetric!"

    α = sqrt.(Δt .* HJ ./ 8 ./ SUN)
    γ = [1 + sqrt(6) / 3, 1 + sqrt(6) / 3, 1 - sqrt(6) / 3, 1 - sqrt(6) / 3]
    η = [sqrt(2 * (3 - sqrt(6))), -sqrt(2 * (3 - sqrt(6))), sqrt(2 * (3 + sqrt(6))), -sqrt(2 * (3 + sqrt(6)))]

    # 预计算每个状态对应的指数值，避免重复计算
    # exp(-α*η) 和 exp(α*η) 对于每个状态值是常数
    # 状态值 s ∈ {1,2,3,4} 对应 model.η 的索引
    exp_αη_neg = [exp(-i * j) for i in α, j in η]  # 大小: length(α) × 4
    exp_αη_pos = [exp(i * j) for i in α, j in η]    # 大小: length(α) × 4
    αη = [i * j for i in α, j in η]

    if div(Nt, 2) % BatchSize == 0
        nodes = collect(0:BatchSize:Nt)
    else
        nodes = vcat(0, reverse(collect(div(Nt, 2)-BatchSize:-BatchSize:1)), collect(div(Nt, 2):BatchSize:Nt), Nt)
    end

    uv = [1 1; 1im -1im] / sqrt(2)
    UV = zeros(ComplexF64, Ns, Ns, size(nnidx, 2))
    for j in axes(nnidx, 2)
        for i in axes(nnidx, 1)
            x, y = nnidx[i, j]
            UV[[x, y], [x, y], j] .= uv
        end
    end

    rng = MersenneTwister(Threads.threadid() + time_ns())
    elements = (1, 2, 3, 4)
    samplers_dict = Dict{UInt8,Random.Sampler}()
    for excluded in elements
        allowed = [i for i in elements if i != excluded]
        samplers_dict[excluded] = Random.Sampler(rng, allowed)
    end

    println("Majorana: $(Lattice) SU$(SUN) size=$(site)  Δt=$(Δt)  Θ=$(Θrelax)+$(Θquench)  U=$(HJ1)--$(HJ2)  Initial=$Initial  flux=$(flux)  opt=$opt  BS=$(BatchSize)  $(Nt)*$(Ns)*$(size(K))")

    return M_VBS_Hubbard_Para_(SUN, Lattice, Ht, HJ1, HJ2, site, Θrelax, Θquench, Ns,
        Nt, realK, BatchSize, Δt, exp_αη_pos, exp_αη_neg, αη, γ, Pt,
        HalfeK, eK, HalfeKinv, eKinv, nnidx, nodes, UV, samplers_dict, flux)

end

mutable struct UpdateBuffer_
    acc::Int64
    uv::Matrix{ComplexF64}      # 2 x 2
    tmp22::Matrix{ComplexF64}   # 2 x 2
    tmp2::Vector{ComplexF64}    # length 2
    r::Matrix{ComplexF64}       # 2 x 2
    Δ::Matrix{ComplexF64}       # 2 x 2
    subidx::Vector{Int64}  # length 2
end

function UpdateBuffer()
    uv = [1 1; 1im -1im] / sqrt(2)
    return UpdateBuffer_(
        0,
        uv,
        Matrix{ComplexF64}(undef, 2, 2),
        Vector{ComplexF64}(undef, 2),
        Matrix{ComplexF64}(undef, 2, 2),
        Matrix{ComplexF64}(undef, 2, 2),
        Vector{Int64}(undef, 2),
    )
end


# ---------------------------------------------------------------------------------------

function PhyBuffer(Ns, NN)
    ns = div(Ns, 2)
    return PhyBuffer_(
        Vector{ComplexF64}(undef, ns),
        Vector{LAPACK.BlasInt}(undef, ns),
        Matrix{ComplexF64}(undef, Ns, Ns),
        Matrix{ComplexF64}(undef, Ns, Ns),
        Array{ComplexF64}(undef, ns, Ns, NN),
        Array{ComplexF64}(undef, Ns, ns, NN),
        Vector{ComplexF64}(undef, Ns),
        Matrix{ComplexF64}(undef, Ns, Ns),
        Matrix{ComplexF64}(undef, Ns, ns),
        Matrix{ComplexF64}(undef, ns, ns),
        Matrix{ComplexF64}(undef, ns, Ns),
        Matrix{ComplexF64}(undef, 2, Ns),
    )
end
# ---------------------------------------------------------------------------------------

function SCEEBuffer(Ns)
    ns = div(Ns, 2)
    return SCEEBuffer_(
        Matrix{ComplexF64}(I, Ns, Ns),
        Vector{ComplexF64}(undef, Ns),
        Vector{ComplexF64}(undef, Ns),
        Matrix{ComplexF64}(undef, 2, Ns),
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
        Matrix{ComplexF64}(undef, Ns, Ns),
        Array{ComplexF64,3}(undef, ns, Ns, NN),
        Array{ComplexF64,3}(undef, Ns, ns, NN),
        Array{ComplexF64,3}(undef, Ns, Ns, NN),
        Array{ComplexF64,3}(undef, Ns, Ns, NN),
    )
end

function AreaBuffer(index)
    nA = length(index)
    return AreaBuffer_(
        index,
        zero(Float64),
        Matrix{ComplexF64}(undef, nA, nA),
        Matrix{ComplexF64}(undef, nA, nA),
        Matrix{ComplexF64}(undef, nA, 2),
        Matrix{ComplexF64}(undef, 2, nA),
        Matrix{ComplexF64}(undef, nA, 2),
        Matrix{ComplexF64}(undef, 2, nA),
        Matrix{ComplexF64}(undef, 2, 2),
        Vector{LAPACK.BlasInt}(undef, nA),
    )
end

# undeveloped 
function DOPBuffer(alpha, index)
    nA = length(index)
    return DOPBuffer_(
        alpha,
        index,
        zero(ComplexF64),
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

