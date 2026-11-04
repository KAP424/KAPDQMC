"""Independent latent-j strata for the existing fixed-half Born ensemble.

Every call draws a fresh auxiliary F. A range stratum includes every j in its
range, sampled from its normalized exact analytic weights; no tail is dropped.
The auxiliary occupation F is never used as the physical postmeasurement state.
"""
module StratifiedBorn
using LinearAlgebra, Random, SHA
include("../BornSampler.jl")
using .BornSampler
export StratifiedSampler, stratum_probability, sample_stratum, sample_fixed_j,
       record_digest, logsumexp, BornSampler

logsumexp(x) = (m=maximum(x); m+log(sum(exp.(x .- m))))

struct StratifiedSampler
    iid::IIDSampler
    log_jweights::Vector{Float64}
    log_Z::Float64
end

function StratifiedSampler(P::AbstractMatrix{<:Real}, beta::Real)
    iid=IIDSampler(P,beta)
    lw=BornSampler.log_jweights(size(iid.Q,2),iid.beta)
    StratifiedSampler(iid,lw,logsumexp(lw))
end

function check_range(s,jlo,jhi)
    a=size(s.iid.Q,2)
    0<=jlo<=jhi<=a || throw(ArgumentError("stratum must satisfy 0 <= jlo <= jhi <= a"))
    nothing
end

function stratum_probability(s::StratifiedSampler,jlo::Integer,jhi::Integer=jlo)
    check_range(s,jlo,jhi)
    exp(logsumexp(view(s.log_jweights,jlo+1:jhi+1))-s.log_Z)
end

# Gumbel-max uses log weights directly, avoiding explicit zeroing of tiny tail
# weights. Sampling and entropy remain finite-precision calculations.
function draw_j(rng,s,jlo,jhi)
    best=-Inf; selected=jlo
    for j in jlo:jhi
        u=rand(rng)
        while u==0.;u=rand(rng);end
        score=s.log_jweights[j+1]-log(-log(u))
        if score>best;best=score;selected=j;end
    end
    selected
end

function sample_stratum(s::StratifiedSampler,jlo::Integer,jhi::Integer=jlo;
                        rng=Random.default_rng())
    check_range(s,jlo,jhi)
    N,a=size(s.iid.Q)
    F=sort!(BornSampler.sample_projection(rng,s.iid.Q))
    j=jlo==jhi ? Int(jlo) : draw_j(rng,s,jlo,jhi)
    Fc=setdiff(1:N,F)
    E=sort!(vcat(F[randperm(rng,a)[1:j]],Fc[randperm(rng,a)[1:a-j]]))
    length(E)==a && length(unique(E))==a && length(intersect(E,F))==j || error("invalid latent sample")
    (;F,E,j)
end

sample_fixed_j(s::StratifiedSampler,j::Integer;kwargs...)=sample_stratum(s,j,j;kwargs...)

record_digest(F,E,j)=bytes2hex(sha256("F:"*join(F,';')*"|E:"*join(E,';')*"|j:"*string(j)))
end
