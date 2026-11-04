"""
Independent sampling of half-filled weak-measurement outcomes for a fixed real
Slater state. This samples p(E) proportional to ||exp(-2beta N_E) Phi(P)||^2,
conditioned on |E| = Nsites/2. It does not sample unrestricted local outcomes.
"""
module BornSampler
using LinearAlgebra, Random
export IIDSampler, sample_measurement, log_normalizer, log_weight, log_probability

function check_beta(beta)
    isfinite(beta) && beta >= 0 || throw(ArgumentError("beta must be finite and nonnegative"))
end

function log_jweights(a::Integer, beta::Real)
    a > 0 || throw(ArgumentError("particle number must be positive"))
    check_beta(beta)
    weights = zeros(Float64, a + 1)
    for j in 1:a
        weights[j + 1] = weights[j] + 2log((a-j+1)/j) - 4beta
    end
    weights
end

"""log sum_{|E|=a} w(E), valid for any rank-a projector on 2a sites."""
function log_normalizer(a::Integer, beta::Real)
    w = log_jweights(a, beta)
    m = maximum(w)
    m + log(sum(exp.(w .- m)))
end

struct IIDSampler
    Q::Matrix{Float64}
    beta::Float64
    jweights::Vector{Float64}
end

function IIDSampler(P::AbstractMatrix{<:Real}, beta::Real)
    check_beta(beta)
    beta = Float64(beta)
    check_beta(beta)
    N, a = size(P)
    N == 2a || throw(ArgumentError("this sampler requires Nsites = 2Nparticles"))
    a > 0 && all(isfinite, P) || throw(ArgumentError("invalid orbital matrix"))
    F = qr(Matrix{Float64}(P))
    d = abs.(diag(F.R))
    minimum(d) > eps(Float64)*max(N,a)*maximum(d) ||
        throw(ArgumentError("P must have full column rank"))
    Q = Matrix(F.Q)[:, 1:a]
    lw = log_jweights(a, beta)
    w = exp.(lw .- maximum(lw))
    w ./= sum(w)
    IIDSampler(Q, Float64(beta), w)
end

function weighted_index(rng, weights)
    total = sum(weights)
    isfinite(total) && total > 0 || error("invalid sampling weights")
    u = rand(rng)*total
    accum = 0.0
    lastpositive = 0
    for i in eachindex(weights)
        weights[i] >= 0 || error("negative sampling weight")
        if weights[i] > 0
            lastpositive = i
            accum += weights[i]
            u < accum && return i
        end
    end
    lastpositive # protects against rounding at the end of the cumulative sum
end

# Projection DPP using a Householder rotation to remove one orbital at each
# selected site. V remains orthonormal on its active columns in exact arithmetic.
function sample_projection(rng, Q::Matrix{Float64})
    V = copy(Q)
    N, a = size(V)
    chosen = Int[]
    weights = zeros(N)
    work = zeros(N)
    for k in a:-1:1
        active = view(V, :, 1:k)
        for r in 1:N
            weights[r] = sum(abs2, view(active, r, :))
        end
        weights[chosen] .= 0.0
        i = weighted_index(rng, weights)
        push!(chosen, i)
        if k > 1
            v = collect(view(active, i, :))
            v[k] += copysign(norm(v), v[k])
            v ./= norm(v)
            mul!(work, active, v)
            BLAS.ger!(-2.0, work, v, active)
            V[i, :] .= 0.0
        end
    end
    chosen
end

"""Draw an independent E; the auxiliary F must NOT replace the original state."""
function sample_measurement(sampler::IIDSampler; rng=Random.default_rng())
    N, a = size(sampler.Q)
    sampler.beta == 0 && return sort!(randperm(rng, N)[1:a])
    F = sample_projection(rng, sampler.Q)
    Fc = setdiff(1:N, F)
    j = weighted_index(rng, sampler.jweights) - 1
    selected_F = F[randperm(rng, a)[1:j]]
    selected_Fc = Fc[randperm(rng, a)[1:a-j]]
    sort!(vcat(selected_F, selected_Fc))
end

function log_weight(sampler::IIDSampler, E::AbstractVector{<:Integer})
    N, a = size(sampler.Q)
    length(E) == a && length(unique(E)) == a && all(r -> 1 <= r <= N, E) ||
        throw(ArgumentError("E must contain Nparticles distinct, valid site indices"))
    filtered = copy(sampler.Q)
    filtered[E, :] .*= exp(-2sampler.beta)
    # QR avoids determinant underflow and subtracting two nearly equal matrices.
    R = qr!(filtered).R
    2sum(log, abs.(diag(R)))
end

log_probability(s::IIDSampler, E) = log_weight(s, E) - log_normalizer(size(s.Q,2), s.beta)

end # module
