"""
Free spinless fermions on the periodic HoneyComb120 lattice, with arbitrary
half-filled m × m configurations. All entropies are natural-log second Renyi
entropies for ONE fermion species. No factor of two is implicit.

Only LinearAlgebra is required. Local orbital order is
`s + 2*(x + m*y)`, x,y=0:m-1, s=1,2. Real-space site order matches Geometry.jl.
The hopping convention is +t, as in nnK_Matrix("HoneyComb120", ...).
"""
module FermionExperiment

using LinearAlgebra

export BlochModel, PreparedPattern, FermionState, prepare_pattern, evolve, measure,
       diagnostics, entropy_strip, entropy_rectangle, correlation_matrix,
       rectangle_indices, occupied_indices, special_pattern, entropy2,
       projector_distance, dense_hamiltonian, pattern_statistics

struct BlochModel
    L::Int
    m::Int
    q::Int
    hamiltonians::Vector{Matrix{ComplexF64}}
    energies::Vector{Vector{Float64}}
    vectors::Vector{Matrix{ComplexF64}}
    ground_energy::Float64
end

"""Construct 2m²-dimensional Bloch Hamiltonians for L divisible by m."""
function BlochModel(L::Integer, m::Integer; hopping=1.0)
    L >= 3 && m >= 1 && L % m == 0 ||
        throw(ArgumentError("require L >= 3, m >= 1 and L % m == 0"))
    q, n = L ÷ m, 2m^2
    hs = Matrix{ComplexF64}[]
    es = Vector{Float64}[]
    vs = Matrix{ComplexF64}[]
    for ky in 0:q-1, kx in 0:q-1
        h = zeros(ComplexF64, n, n)
        for y in 0:m-1, x in 0:m-1
            a = 1 + 2(x + m*y)
            # A(x,y) couples to B(x,y), B(x+1,y), and B(x,y-1).
            for (dx, dy) in ((0, 0), (1, 0), (0, -1))
                bx, by = mod(x + dx, m), mod(y + dy, m)
                rx, ry = fld(x + dx, m), fld(y + dy, m)
                b = 2 + 2(bx + m*by)
                value = hopping * cis(2pi * (kx*rx + ky*ry) / q)
                h[a,b] += value
                h[b,a] += conj(value)
            end
        end
        eig = eigen(Hermitian(h))
        push!(hs, h); push!(es, eig.values); push!(vs, eig.vectors)
    end
    e0 = sum(sum(e[1:m^2]) for e in es)
    return BlochModel(L, m, q, hs, es, vs, e0)
end

struct PreparedPattern
    model::BlochModel
    occupied::BitVector
    F::Vector{Int}
    E::Vector{Int}
    coefficients::Vector{Matrix{ComplexF64}}
    gaps::Vector{Matrix{Float64}}
    pivots::Vector{Vector{Float64}}
    accessible_energy::Float64
    max_reconstruction_error::Float64
    min_pivot_sv::Float64
    tol::Float64
end

"""
    prepare_pattern(model, occupied; tol=1e-10, degeneracy_tol=1e-11)

Build an energy-adapted right basis of the initial Slater orbitals. Columns
whose first nonzero spectral component has energy e_j can be scaled by
exp(theta*e_j) independently without changing their span. This removes
exponential dynamic range at finite theta while preserving every decaying
component. Entries known to vanish by numerical-rank decisions are set to zero.

The absolute singular-value threshold `tol` defines the numerical support of
the initial state: genuine components smaller than tol are also discarded.
`max_reconstruction_error` records the resulting theta=0 orbital error. This
method does NOT replace finite theta by an equilibrium state.
"""
function prepare_pattern(model::BlochModel, occupied; tol=1e-10, degeneracy_tol=1e-11)
    n, p = 2model.m^2, model.m^2
    length(occupied) == n || throw(DimensionMismatch("pattern must have 2m² entries"))
    all(x -> x == 0 || x == 1, occupied) || throw(ArgumentError("pattern must be binary"))
    occ = BitVector(occupied)
    count(occ) == p || throw(ArgumentError("pattern must be half filled"))
    0 < tol < 1 || throw(ArgumentError("tol must be between 0 and 1"))
    degeneracy_tol >= 0 || throw(ArgumentError("degeneracy_tol must be nonnegative"))
    F, E = findall(occ), findall(.!occ)
    coefficients = Matrix{ComplexF64}[]
    gaps = Matrix{Float64}[]
    pivots = Vector{Float64}[]
    accessible_energy, reconstruction_error, min_sv = 0.0, 0.0, Inf
    for k in eachindex(model.energies)
        energy, v = model.energies[k], model.vectors[k]
        c = Matrix(v[F,:]')
        right = zeros(ComplexF64, p, p)
        starts = Int[]
        pivot = Float64[]
        clustered = copy(energy)
        used, first = 0, 1
        while first <= n
            last = first
            while last < n && abs(energy[last+1] - energy[first]) <= degeneracy_tol
                last += 1
            end
            level = sum(energy[first:last]) / (last-first+1)
            clustered[first:last] .= level
            if used < p
                projected = copy(c[first:last,:])
                previous = view(right, :, 1:used)
                for _ in 1:2
                    projected -= (projected * previous) * previous'
                end
                decomp = svd(projected)
                added = count(>(tol), decomp.S)
                added <= p-used || error("inconsistent numerical rank; increase tol")
                if added > 0
                    newright = decomp.V[:,1:added]
                    for _ in 1:2
                        newright -= previous * (previous' * newright)
                    end
                    newright = Matrix(qr(newright).Q)[:,1:added]
                    right[:,used+1:used+added] = newright
                    append!(starts, fill(first, added))
                    append!(pivot, fill(level, added))
                    min_sv = min(min_sv, decomp.S[added])
                    used += added
                end
            end
            first = last + 1
        end
        used == p || error("initial rank lost at tolerance $tol")
        transformed = c * right
        cleaned = copy(transformed)
        gap = zeros(n, p)
        for j in 1:p
            cleaned[1:starts[j]-1,j] .= 0
            gap[:,j] = max.(0.0, clustered .- pivot[j])
        end
        reconstruction_error = max(reconstruction_error, norm(cleaned-transformed))
        accessible_energy += sum(pivot)
        push!(coefficients, cleaned); push!(gaps, gap); push!(pivots, pivot)
    end
    return PreparedPattern(model, occ, F, E, coefficients, gaps, pivots,
        accessible_energy, reconstruction_error, min_sv, Float64(tol))
end

struct FermionState
    prepared::PreparedPattern
    theta::Float64
    beta::Float64
    orbitals::Vector{Matrix{ComplexF64}}
end

"""Normalized finite-time state, or initial-state-accessible limit at theta=Inf."""
function evolve(prepared::PreparedPattern, theta::Real; beta::Real=0.0)
    theta >= 0 || throw(ArgumentError("theta must be nonnegative"))
    model, p = prepared.model, length(prepared.F)
    orbitals = Matrix{ComplexF64}[]
    for k in eachindex(model.energies)
        if theta == 0
            q = zeros(ComplexF64, 2p, p)
            q[prepared.F,:] = Matrix{ComplexF64}(I, p, p)
        else
            weights = isinf(theta) ? Float64.(prepared.gaps[k] .== 0) :
                exp.(-theta .* prepared.gaps[k])
            raw = model.vectors[k] * (prepared.coefficients[k] .* weights)
            q = Matrix(qr(raw).Q)[:,1:p]
        end
        push!(orbitals, q)
    end
    state = FermionState(prepared, Float64(theta), 0.0, orbitals)
    return beta == 0 ? state : measure(state, beta)
end

"""Apply exp(-2beta*N_E), then normalize each Slater determinant by thin QR."""
function measure(state::FermionState, beta::Real)
    beta >= 0 || throw(ArgumentError("beta must be nonnegative"))
    beta == 0 && return state
    p, pre = length(state.prepared.F), state.prepared
    orbitals = Matrix{ComplexF64}[]
    for q in state.orbitals
        raw = copy(q)
        raw[pre.E,:] .*= exp(-2beta)
        minimum(svdvals(raw)) > 100eps(Float64) ||
            error("measurement loses numerical rank; initial configuration overlap is too small")
        push!(orbitals, Matrix(qr(raw).Q)[:,1:p])
    end
    return FermionState(pre, state.theta, state.beta+Float64(beta), orbitals)
end

"""Energy and residual are total single-species quantities, not divided by Ns."""
function diagnostics(state::FermionState)
    pre, model = state.prepared, state.prepared.model
    energy, residual2, logoverlap, empty, projerr = 0.0, 0.0, 0.0, 0.0, 0.0
    minsv = 1.0
    for k in eachindex(state.orbitals)
        q = state.orbitals[k]
        hq = model.hamiltonians[k] * q
        energy += real(tr(q' * hq))
        residual2 += sum(abs2, hq - q*(q'*hq))
        s = svdvals(q[pre.F,:])
        minsv = min(minsv, s[end])
        logoverlap += 2sum(log, s)
        empty += sum(abs2, q[pre.E,:])
        projerr = max(projerr, norm(q'*q-I))
    end
    return (energy=energy, excess_energy=energy-model.ground_energy,
        accessible_excess=pre.accessible_energy-model.ground_energy,
        residual=sqrt(residual2), min_initial_sv=minsv,
        log_initial_overlap2=logoverlap, empty_occupancy=empty,
        particle_number=Float64(model.L^2), projector_error=projerr,
        reconstruction_error=pre.max_reconstruction_error, min_pivot_sv=pre.min_pivot_sv)
end

function projector_distance(a::FermionState, b::FermionState; normalized=true)
    ma, mb = a.prepared.model, b.prepared.model
    (ma.L,ma.m) == (mb.L,mb.m) || throw(ArgumentError("models must have same L and m"))
    value = sqrt(sum(sum(abs2, x*x'-y*y') for (x,y) in zip(a.orbitals,b.orbitals)))
    return normalized ? value/sqrt(ma.L^2) : value
end

"""Special pattern: +1 means A occupied/B empty; -1 means B occupied/A empty."""
function special_pattern(config::AbstractMatrix)
    size(config,1) == size(config,2) || throw(ArgumentError("config must be square"))
    all(x -> x == 1 || x == -1, config) || throw(ArgumentError("config entries must be ±1"))
    m = size(config,1)
    return BitVector([config[x+1,y+1] == (s == 1 ? 1 : -1)
        for y in 0:m-1 for x in 0:m-1 for s in 1:2])
end

function occupied_indices(pre::PreparedPattern)
    model = pre.model
    return [s+2*((x-1)+model.L*(y-1)) for y in 1:model.L for x in 1:model.L for s in 1:2
        if pre.occupied[s+2*(mod(x-1,model.m)+model.m*mod(y-1,model.m))]]
end

function pattern_statistics(occupied)
    n = length(occupied)
    iseven(n) || throw(ArgumentError("pattern size must be even"))
    double = count(i -> occupied[2i-1] && occupied[2i], 1:n÷2)
    empty = count(i -> !occupied[2i-1] && !occupied[2i], 1:n÷2)
    return (occupied=count(occupied), double_cells=double, empty_cells=empty,
        special=double == 0 && empty == 0,
        sublattice_imbalance=sum(occupied[1:2:end])-sum(occupied[2:2:end]))
end

function rectangle_indices(L::Integer; width=L÷3, height=2L÷3, x0=1, y0=1)
    1 <= x0 <= x0+width-1 <= L && 1 <= y0 <= y0+height-1 <= L ||
        throw(ArgumentError("rectangle must lie inside lattice"))
    return [s+2*(x-1+L*(y-1)) for y in y0:y0+height-1 for x in x0:x0+width-1 for s in 1:2]
end

function site_coordinates(model::BlochModel, site::Integer)
    1 <= site <= 2model.L^2 || throw(BoundsError())
    s, cell = mod1(site,2), (site-1)÷2
    x, y = mod(cell,model.L), cell÷model.L
    a = s+2*(mod(x,model.m)+model.m*mod(y,model.m))
    return a, x÷model.m, y÷model.m
end

# Kernel[a,b,dx+1,dy+1] is the real-space covariance between supercells.
# Separable explicit Fourier transforms avoid a non-standard FFT dependency.
function correlation_kernel(state::FermionState)
    model = state.prepared.model
    q, n = model.q, 2model.m^2
    cs = [u*u' for u in state.orbitals]
    fourier = [cis(2pi*r*k/q)/q for r in 0:q-1, k in 0:q-1]
    kernel = Array{ComplexF64}(undef, n,n,q,q)
    spectrum = Matrix{ComplexF64}(undef,q,q)
    for b in 1:n, a in 1:n
        for ky in 0:q-1, kx in 0:q-1
            spectrum[kx+1,ky+1] = cs[1+kx+q*ky][a,b]
        end
        realspace = fourier*spectrum*transpose(fourier)
        kernel[a,b,:,:] = realspace
    end
    return kernel
end

"""Reconstruct C[i,j]=<c†_j c_i> on arbitrary specified real-space indices."""
function correlation_matrix(state::FermionState, indices=1:2state.prepared.model.L^2)
    model = state.prepared.model
    kernel = correlation_kernel(state)
    coords = [site_coordinates(model,i) for i in indices]
    n = length(indices)
    c = Matrix{ComplexF64}(undef,n,n)
    for j in 1:n, i in 1:n
        a,x,y = coords[i]; b,u,v = coords[j]
        c[i,j] = kernel[a,b,mod(x-u,model.q)+1,mod(y-v,model.q)+1]
    end
    return Matrix(Hermitian((c+c')/2))
end

"""S₂ = -Σ log(λ²+(1-λ)²); eigenvalues outside numerical [0,1] fail loudly."""
function entropy2(c::AbstractMatrix; copies=1)
    # Real hopping and a real configuration imply a real full-space covariance.
    # Individual ky strip blocks can still be complex and retain Hermitian LAPACK.
    values = eltype(c) <: Complex && maximum(abs,imag.(c)) <= 2e-12 ?
        eigvals(Symmetric(real.(c))) : eigvals(Hermitian(c))
    minimum(values) >= -2e-9 && maximum(values) <= 1+2e-9 ||
        error("correlation eigenvalues outside [0,1]")
    values = clamp.(values,0.0,1.0)
    return -copies*sum(log1p.(-2 .* values .* (1 .- values)))
end

"""
Strip S₂ by Fourier decomposition along its uncut direction. orientation=:x
means finite x width and full y; :y means finite y width and full x. The cut
need not align with supercell boundaries. `x0` denotes the start on the cut axis.
"""
function entropy_strip(state::FermionState; width=state.prepared.model.L÷3,
        x0=1, orientation=:x, copies=1)
    model = state.prepared.model
    m, q, n = model.m, model.q, 2model.m^2
    orientation in (:x,:y) || throw(ArgumentError("orientation must be :x or :y"))
    1 <= x0 <= x0+width-1 <= model.L || throw(ArgumentError("invalid strip"))
    cs = [u*u' for u in state.orbitals]
    coords = Tuple{Int,Int}[]
    for internal in 0:m-1, pos in x0-1:x0+width-2, s in 1:2
        x,y = orientation == :x ? (mod(pos,m),internal) : (internal,mod(pos,m))
        push!(coords,(s+2*(x+m*y),pos÷m))
    end
    d = length(coords)
    c = Matrix{ComplexF64}(undef,d,d)
    kernel = zeros(ComplexF64,n,n,q)
    entropy = 0.0
    for parallel in 0:q-1
        fill!(kernel,0)
        for perpendicular in 0:q-1
            kx,ky = orientation == :x ? (perpendicular,parallel) : (parallel,perpendicular)
            ck = cs[1+kx+q*ky]
            for r in 0:q-1
                @views kernel[:,:,r+1] .+= (cis(2pi*r*perpendicular/q)/q) .* ck
            end
        end
        for j in 1:d, i in 1:d
            a,r = coords[i]; b,t = coords[j]
            c[i,j] = kernel[a,b,mod(r-t,q)+1]
        end
        entropy += entropy2(c; copies=copies)
    end
    return entropy
end

function entropy_rectangle(state::FermionState; width=state.prepared.model.L÷3,
        height=2state.prepared.model.L÷3, x0=1, y0=1, copies=1)
    model = state.prepared.model
    if height == model.L && y0 == 1
        return entropy_strip(state; width=width,x0=x0,copies=copies)
    elseif width == model.L && x0 == 1
        return entropy_strip(state; width=height,x0=y0,orientation=:y,copies=copies)
    end
    indices = rectangle_indices(model.L;width=width,height=height,x0=x0,y0=y0)
    return entropy2(correlation_matrix(state,indices);copies=copies)
end

"""Small-system reference hopping matrix (large studies should use BlochModel)."""
function dense_hamiltonian(L::Integer; hopping=1.0)
    L >= 3 || throw(ArgumentError("require L >= 3"))
    h = zeros(Float64,2L^2,2L^2)
    for y in 0:L-1, x in 0:L-1
        a = 1+2*(x+L*y)
        for (dx,dy) in ((0,0),(1,0),(0,-1))
            b = 2+2*(mod(x+dx,L)+L*mod(y+dy,L))
            h[a,b] = h[b,a] = hopping
        end
    end
    return h
end

end # module
