
function SO3Initial_Pt!(Lattice, Initial, Pt, K)
    Ns = size(K, 1)
    if Initial == "H0"
        KK = copy(K)
        μ = 1e-5
        KK .+= μ * diagm(repeat([-1, -1, -1, 1, 1, 1], div(Ns, 6)))
        E, V = LAPACK.syevd!('V', 'L', KK)
        Pt .= V[:, 1:div(Ns, 2)]
    elseif Initial == "V"
        if Lattice == "SQUARE90" || Lattice == "HoneyComb120" || Lattice == "HoneyComb60"
            count = 1
            for i in 1:div(Ns, 6)
                Pt[(i-1)*6+1, count] = 1
                Pt[(i-1)*6+2, count+1] = 1
                Pt[(i-1)*6+3, count+2] = 1
                count += 3
            end
        elseif Lattice == "SQUARE45"
            count = 1
            for i in 1:Ns
                x, y = i_xy(Lattice, site, i)
                if (x + y) % 2 == 1
                    Pt[i, count] = 1
                    count += 1
                    if count > div(Ns, 2)
                        break
                    end
                end
            end
        end
    elseif Initial == "HJ"

    else
        error("Initial state $Initial not supported!")
    end

end

function so3Tindex(site, bond)
    if bond == 1
        return (site - 1) * 3 + 2, (site - 1) * 3 + 3
    elseif bond == 2
        return (site - 1) * 3 + 1, (site - 1) * 3 + 3
    elseif bond == 3
        return (site - 1) * 3 + 1, (site - 1) * 3 + 2
    end
end

function so3Tindex_F(Lattice, site)
    if occursin("HoneyComb", Lattice)
        Ns = 2 * prod(site)
    end
    nnidx = fill((0, 0), Ns, 3)

    for i in 1:Ns
        for bond in 1:3
            nnidx[i, bond] = so3Tindex(i, bond)
        end
    end
    return nnidx
end

function nnK_Matrix4so3(Lattice, site, flux=0.0, opt="xy")
    if Lattice == "SQUARE90"
        error("Lattice $Lattice not supported!")
    elseif Lattice == "HoneyComb120"
        kk = nnK_Matrix(Lattice, site, flux=flux, opt=opt)

        N = size(kk, 1)
        K = zeros(ComplexF64, 3 * N, 3 * N)

        K[1:3:end, 1:3:end] = kk
        K[2:3:end, 2:3:end] = kk
        K[3:3:end, 3:3:end] = kk

    else
        error("Lattice $Lattice not supported!")
    end
    return K
end


struct SO3_Hubbard_Para_
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
    α::Vector{Float64}
    γ::Vector{Float64}
    η::Vector{Float64}
    Pt::Array{ComplexF64,2}
    HalfeK::Array{ComplexF64,2}
    eK::Array{ComplexF64,2}
    HalfeKinv::Array{ComplexF64,2}
    eKinv::Array{ComplexF64,2}
    bondidx::Matrix{Tuple{Int64,Int64}}
    nodes::Vector{Int64}
    UV::Array{ComplexF64,3}
    samplers_dict::Dict{UInt8,Random.Sampler}
    flux::Float64
end


function SO3_Hubbard_Para(; Ht, HJ1, HJ2, Δt, Θrelax, Θquench, Lattice::String, site, BatchSize, Initial::String, flux=0.0, opt="xy")
    K = nnK_Matrix4so3(Lattice, site, flux, opt)
    Ns = size(K, 1)

    E, V = LAPACK.syevd!('V', 'L', -Ht .* K[:, :])
    if abs(E[div(Ns, 2)] - E[div(Ns, 2)+1]) > 1e-10
        @warn "Warning: The non-interacting system may be gapped!"
    end
    HalfeK = V * Diagonal(exp.(-Δt .* E ./ 2)) * V'
    eK = V * Diagonal(exp.(-Δt .* E)) * V'
    HalfeKinv = V * Diagonal(exp.(Δt .* E ./ 2)) * V'
    eKinv = V * Diagonal(exp.(Δt .* E)) * V'

    Pt = zeros(ComplexF64, Ns, div(Ns, 2))
    SO3Initial_Pt!(Lattice, Initial, Pt, K)
    @assert norm(Pt' * Pt - I(div(Ns, 2))) < 1e-10 "Pt is not unitary!"

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

    α = sqrt.(Δt .* HJ ./ 2)
    γ = [1 + sqrt(6) / 3, 1 + sqrt(6) / 3, 1 - sqrt(6) / 3, 1 - sqrt(6) / 3]
    η = [sqrt(2 * (3 - sqrt(6))), -sqrt(2 * (3 - sqrt(6))), sqrt(2 * (3 + sqrt(6))), -sqrt(2 * (3 + sqrt(6)))]

    if div(Nt, 2) % BatchSize == 0
        nodes = collect(0:BatchSize:Nt)
    else
        nodes = vcat(0, reverse(collect(div(Nt, 2)-BatchSize:-BatchSize:1)), collect(div(Nt, 2):BatchSize:Nt), Nt)
    end

    bondidx = so3Tindex_F(Lattice, site)
    UV = zeros(ComplexF64, Ns, Ns, 3)
    uv = [1 1; -1im 1im] / sqrt(2)
    for ns in 1:div(Ns, 3)
        UV[[3 * (ns - 1) + 2, 3 * (ns - 1) + 3], [3 * (ns - 1) + 2, 3 * (ns - 1) + 3], 1] .= uv
        UV[[3 * (ns - 1) + 1, 3 * (ns - 1) + 3], [3 * (ns - 1) + 1, 3 * (ns - 1) + 3], 2] .= uv
        UV[[3 * (ns - 1) + 2, 3 * (ns - 1) + 1], [3 * (ns - 1) + 2, 3 * (ns - 1) + 1], 3] .= uv
    end
    UV[1:3:end, 1:3:end, 1] .= I(div(Ns, 3))
    UV[2:3:end, 2:3:end, 2] .= I(div(Ns, 3))
    UV[3:3:end, 3:3:end, 3] .= I(div(Ns, 3))

    rng = MersenneTwister(Threads.threadid() + time_ns())
    elements = (1, 2, 3, 4)
    samplers_dict = Dict{UInt8,Random.Sampler}()
    for excluded in elements
        allowed = [i for i in elements if i != excluded]
        samplers_dict[excluded] = Random.Sampler(rng, allowed)
    end

    return SO3_Hubbard_Para_(Lattice, Ht, HJ1, HJ2, site, Θrelax, Θquench, Ns,
        Nt, K, BatchSize, Δt, α, γ, η, Pt,
        HalfeK, eK, HalfeKinv, eKinv, bondidx, nodes, UV, samplers_dict, flux)

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
    uv = [1 1; -1im 1im] / sqrt(2)
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
# Buffers for SCEE workflow

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
        Matrix{ComplexF64}(undef, nA, 2),
        Matrix{ComplexF64}(undef, 2, nA),
        Matrix{ComplexF64}(undef, nA, 2),
        Matrix{ComplexF64}(undef, 2, nA),
        Matrix{ComplexF64}(undef, 2, 2),
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
        Matrix{ComplexF64}(undef, nA, 2),
        Matrix{ComplexF64}(undef, 2, nA),
        Matrix{ComplexF64}(undef, nA, 2),
        Matrix{ComplexF64}(undef, 2, nA),
        Matrix{ComplexF64}(undef, 2, 2),
        Vector{LAPACK.BlasInt}(undef, nA),
    )
end


