"""
Optional exact momentum folding for real-space covariance matrices.
Include fermion_experiment.jl first. The locked FE implementation is unchanged.
This uses C(-k)=conj(C(k)), not a sublattice or half-spectrum entropy symmetry.
"""
module BetaPatternsStripExtension
using LinearAlgebra
import ..FermionExperiment as FE
export entropy_strip_folded

function strip_block!(c, kernel, cs, coords, fourier, parallel, q, orientation)
    fill!(kernel, 0)
    for perpendicular in 0:q-1
        kx, ky = orientation == :x ? (perpendicular, parallel) : (parallel, perpendicular)
        ck = cs[1+kx+q*ky]
        for r in 0:q-1
            @views kernel[:,:,r+1] .+= fourier[r+1,perpendicular+1] .* ck
        end
    end
    for j in axes(c,2), i in axes(c,1)
        a,r = coords[i]; b,t = coords[j]
        c[i,j] = kernel[a,b,mod(r-t,q)+1]
    end
    return c
end

"""
Strip entropy from p=0:floor(q/2), with multiplicity two except self-inverse p.
Before folding, every small Bloch projector is checked against its (-kx,-ky)
partner. One non-self-inverse strip pair is also reconstructed independently
and checked. This validation adds no partner eigendecomposition.

For real hopping, real initial occupied orbitals and real diagonal measurement,
the full-space covariance is real. Its partial Fourier blocks obey
  C_strip(-p) = conj(C_strip(p)),
even for a cut not aligned with the occupation supercell. Their spectra and
Renyi entropies are identical. No equality of x/y strips is assumed.

with_checks=true returns (entropy, checks), otherwise just the scalar entropy.
Validation failures throw rather than silently discarding complex structure.
"""
function entropy_strip_folded(state::FE.FermionState; width=state.prepared.model.L÷3,
        x0=1, orientation=:x, copies=1, with_checks=false,
        conjugacy_atol=2e-10, block_atol=2e-10)
    model = state.prepared.model
    m, q, n = model.m, model.q, 2model.m^2
    orientation in (:x,:y) || throw(ArgumentError("orientation must be :x or :y"))
    1 <= x0 <= x0+width-1 <= model.L || throw(ArgumentError("invalid strip"))
    conjugacy_atol > 0 && block_atol > 0 || throw(ArgumentError("tolerances must be positive"))
    cs = [u*u' for u in state.orbitals]
    max_bloch_conjugacy = 0.0
    for ky in 0:q-1, kx in 0:q-1
        a = cs[1+kx+q*ky]
        b = cs[1+mod(-kx,q)+q*mod(-ky,q)]
        for index in eachindex(a)
            max_bloch_conjugacy = max(max_bloch_conjugacy, abs(b[index]-conj(a[index])))
        end
    end
    max_bloch_conjugacy <= conjugacy_atol || error("Bloch covariance violates real-space conjugacy: $max_bloch_conjugacy")
    coords = Tuple{Int,Int}[]
    for internal in 0:m-1, pos in x0-1:x0+width-2, s in 1:2
        x,y = orientation == :x ? (mod(pos,m),internal) : (internal,mod(pos,m))
        push!(coords,(s+2*(x+m*y),pos÷m))
    end
    d = length(coords)
    c = Matrix{ComplexF64}(undef,d,d)
    kernel = zeros(ComplexF64,n,n,q)
    fourier = [cis(2pi*r*k/q)/q for r in 0:q-1, k in 0:q-1]
    # An independent opposite-parallel block catches convention/index mistakes.
    # This checks the Fourier matrices, while the all-k check above covers all pairs.
    probe_parallel = q > 2 ? 1 : 0
    strip_block!(c,kernel,cs,coords,fourier,probe_parallel,q,orientation)
    probe = copy(c)
    partner_parallel = mod(-probe_parallel,q)
    strip_block!(c,kernel,cs,coords,fourier,partner_parallel,q,orientation)
    max_probe_conjugacy = maximum(abs,c-conj.(probe))
    max_probe_conjugacy <= block_atol || error("Strip covariance violates opposite-momentum conjugacy: $max_probe_conjugacy")

    entropy = 0.0
    entropy_evaluations, total_weight, trim_count = 0, 0, 0
    max_trim_imaginary = 0.0
    for parallel in 0:q÷2
        if parallel == probe_parallel
            copyto!(c,probe)
        else
            strip_block!(c,kernel,cs,coords,fourier,parallel,q,orientation)
        end
        self_inverse = mod(-parallel,q) == parallel
        weight = self_inverse ? 1 : 2
        if self_inverse
            trim_count += 1
            max_trim_imaginary = max(max_trim_imaginary, maximum(abs,imag.(c)))
        end
        entropy += weight * FE.entropy2(c;copies)
        entropy_evaluations += 1
        total_weight += weight
    end
    total_weight == q || error("Momentum folding multiplicities do not sum to q")
    max_trim_imaginary <= block_atol || error("Self-inverse strip momentum is not real: $max_trim_imaginary")
    checks = (method="conjugate_momentum_fold",q=q,block_dimension=d,
              entropy_evaluations=entropy_evaluations,original_entropy_evaluations=q,
              total_weight=total_weight,trim_count=trim_count,
              max_bloch_conjugacy=max_bloch_conjugacy,max_probe_conjugacy=max_probe_conjugacy,
              max_trim_imaginary=max_trim_imaginary,probe_parallel=probe_parallel,
              probe_partner=partner_parallel,conjugacy_atol=conjugacy_atol,block_atol=block_atol)
    return with_checks ? (entropy,checks) : entropy
end
end
