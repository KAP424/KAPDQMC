"""
Exhaustive Born-trajectory Renyi-2 entropy for a half-filled pure Slater state.
No Monte Carlo and no truncation of the outcome space. Arithmetic is Float64
(ComplexF64 for complex orbitals), rather than symbolic or arbitrary precision.
"""
module ExactBorn
using LinearAlgebra
export born_trajectory_entropy

logsumexp(x) = (m = maximum(x); m + log(sum(exp.(x .- m))))

function fixed_log_normalizer(a, beta)
    lw = zeros(a+1)
    for j in 1:a
        lw[j+1] = lw[j] + 2log((a-j+1)/j) - 4beta
    end
    logsumexp(lw)
end

function foreach_subset(f, N, k)
    E = Vector{Int}(undef,k)
    function visit(pos, first)
        pos > k && return f(E)
        for site in first:N-k+pos
            E[pos] = site
            visit(pos+1,site+1)
        end
        nothing
    end
    visit(1,1)
end

function parse_regions(regions, N)
    if regions isa NamedTuple || regions isa AbstractDict
        names = Symbol.(collect(keys(regions)))
        subsets = [collect(r) for r in values(regions)]
    elseif regions isa AbstractVector{<:Integer}
        names = [:region]
        subsets = [collect(regions)]
    else
        subsets = [collect(r) for r in regions]
        names = [Symbol("region_",j) for j in eachindex(subsets)]
    end
    isempty(names) && throw(ArgumentError("at least one entropy region is required"))
    length(unique(names)) == length(names) || throw(ArgumentError("region names must be distinct"))
    for r in subsets
        all(i -> i isa Integer && 1 <= i <= N, r) ||
            throw(ArgumentError("region site indices must be integers in 1:Nsites"))
        length(unique(r)) == length(r) || throw(ArgumentError("a region may not repeat a site"))
    end
    names, [Int.(r) for r in subsets]
end

function regional_entropy(Q, region, N)
    isempty(region) && return 0.0
    length(region) == N && return 0.0
    # The filtered state is pure, so use the smaller region/complement.
    idx = length(region) > N÷2 ? setdiff(1:N,region) : region
    V = Q[idx,:]
    nu = eigvals(Hermitian(V*V'))
    minimum(nu) >= -1e-10 && maximum(nu) <= 1+1e-10 ||
        error("correlation spectrum outside [0,1]; filtered QR is not reliable")
    nu = clamp.(nu,0.0,1.0)
    -sum(log1p.(-2 .* nu .* (1 .- nu)))
end

"""
    born_trajectory_entropy(P, beta, regions;
        ensemble=:fixed_half, keep_trajectories=true,
        max_configs=1_000_000, normalization_tolerance=1e-9)

Enumerate EVERY measurement outcome in the selected ensemble and compute
`sum_E probability(E) * S2(region | E)` (natural logarithms).

* `P` contains the occupied orbitals of one pure Slater state. It may be
  nonorthogonal or complex, but must have full column rank and Nsites=2Nparticles.
* `regions` is e.g. `(A=idxA, B=idxB, C=idxC)`, a single site-index vector,
  or a collection of region vectors. Regions may overlap or be empty/full.
* `:fixed_half` enumerates binomial(2a,a) suppressed sets E of size a. It matches
  Free_M.jl's swap sampler: p(E)=||exp(-2beta N_E)Psi||^2 / Z_a.
* `:full_local` enumerates 2^(2a) sets, including empty/full E, for the complete
  local POVM K_+/-=exp(+/-beta(n-1/2))/sqrt(2cosh(beta)). It is a DIFFERENT
  ensemble and must not be compared to the fixed-size swap sampler.

Returns the mean entropy, trajectory variance/covariance, mean purity and its
negative logarithm (a different observable), normalization checks, and, by
default, the E list, log weights, probabilities and each trajectory's entropy.
`keep_trajectories=false` omits those large arrays from the return value; this
implementation still retains scalar weights/entropies internally for summation.
The BigInt outcome count is checked against `max_configs` BEFORE allocation.
No invalid/low-weight trajectory is silently discarded.
"""
function born_trajectory_entropy(P::AbstractMatrix, beta::Real, regions;
        ensemble::Symbol=:fixed_half, keep_trajectories::Bool=true,
        max_configs::Integer=1_000_000, normalization_tolerance::Real=1e-9)
    ensemble in (:fixed_half,:full_local) ||
        throw(ArgumentError("ensemble must be :fixed_half or :full_local"))
    isfinite(beta) && beta >= 0 || throw(ArgumentError("beta must be finite and nonnegative"))
    beta = Float64(beta)
    isfinite(beta) || throw(ArgumentError("beta is outside Float64 range"))
    N,a = size(P)
    a > 0 && N == 2a || throw(ArgumentError("requires Nsites = 2Nparticles > 0"))
    max_configs > 0 || throw(ArgumentError("max_configs must be positive"))
    isfinite(normalization_tolerance) && normalization_tolerance > 0 ||
        throw(ArgumentError("normalization_tolerance must be positive and finite"))
    nbig = ensemble == :fixed_half ? binomial(big(N),a) : big(2)^N
    nbig <= min(big(max_configs),big(typemax(Int))) ||
        throw(ArgumentError("exact enumeration requires $nbig configurations; max_configs=$max_configs"))
    nconfigs = Int(nbig)
    names, subsets = parse_regions(regions,N)
    all(isfinite,P) || throw(ArgumentError("orbital matrix contains nonfinite entries"))
    T = eltype(P) <: Real ? Float64 : ComplexF64
    F0 = qr(Matrix{T}(P))
    d0 = abs.(diag(F0.R))
    minimum(d0) > eps(Float64)*N*maximum(d0) ||
        throw(ArgumentError("P must have numerically full column rank"))
    Q0 = Matrix(F0.Q)[:,1:a]
    attenuation = exp(-2beta)
    attenuation > 0 || throw(ArgumentError("filter underflows Float64; use a higher-precision implementation"))

    expected_lw_sum = ensemble == :fixed_half ? fixed_log_normalizer(a,beta) :
        a*log(2.0)+a*log1p(exp(-4beta))
    log_weights = Vector{Float64}(undef,nconfigs)
    raw_log_probabilities = similar(log_weights)
    entropies = Matrix{Float64}(undef,nconfigs,length(subsets))
    configurations = keep_trajectories ? Vector{Vector{Int}}(undef,nconfigs) : nothing
    row = 0
    function measure(E)
        row += 1
        filtered = copy(Q0)
        filtered[E,:] .*= attenuation
        F = qr!(filtered)
        d = abs.(diag(F.R))
        all(x -> isfinite(x) && x>0,d) ||
            error("filtered orbitals lost rank at outcome $row; no trajectory was dropped")
        lw = 2sum(log,d)
        log_weights[row] = lw
        raw_log_probabilities[row] = ensemble == :fixed_half ? lw-expected_lw_sum :
            lw+2beta*(length(E)-a)-2a*log1p(exp(-2beta))
        Q = Matrix(F.Q)[:,1:a]
        for (j,region) in enumerate(subsets)
            entropies[row,j] = regional_entropy(Q,region,N)
        end
        keep_trajectories && (configurations[row] = copy(E))
        nothing
    end
    sizes = ensemble == :fixed_half ? (a:a) : (0:N)
    for k in sizes
        foreach_subset(measure,N,k)
    end
    row == nconfigs || error("enumeration count mismatch")

    lw_sum = logsumexp(log_weights)
    log_mass = logsumexp(raw_log_probabilities)
    normalization_error = abs(expm1(log_mass))
    weight_sum_error = abs(expm1(lw_sum-expected_lw_sum))
    max(normalization_error,weight_sum_error) <= normalization_tolerance ||
        error("normalization check failed: Born mass error=$normalization_error, raw weight sum error=$weight_sum_error")
    # Remove only roundoff from the already-checked analytic normalization.
    log_probabilities = raw_log_probabilities .- log_mass
    probabilities = exp.(log_probabilities)
    mean_entropy = vec(transpose(probabilities)*entropies)
    mean_purity = vec(transpose(probabilities)*exp.(-entropies))
    centered = entropies .- transpose(mean_entropy)
    covariance = Matrix(Hermitian(centered'*(probabilities .* centered)))
    return (; ensemble,beta,nsites=N,nparticles=a,nconfigurations=nconfigs,
        region_names=names,regions=subsets,mean_entropy,mean_purity,
        annealed_entropy=-log.(mean_purity),entropy_covariance=covariance,
        normalization_error,weight_sum_error,log_weight_sum=lw_sum,
        expected_log_weight_sum=expected_lw_sum,
        probability_mass_before_normalization=exp(log_mass),
        probability_sum=sum(probabilities),
        configurations,
        log_weights=keep_trajectories ? log_weights : nothing,
        log_probabilities=keep_trajectories ? log_probabilities : nothing,
        probabilities=keep_trajectories ? probabilities : nothing,
        entropies=keep_trajectories ? entropies : nothing)
end
end # module
