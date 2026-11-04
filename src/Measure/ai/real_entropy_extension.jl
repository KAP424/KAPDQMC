"""
Optional real-covariance entropy implementation; the locked FE module is unchanged.
Include fermion_experiment.jl first, then this file. Only applicable to a state
whose full-space covariance is real to the explicitly reported tolerance.
"""
module RealEntropyExtension
using LinearAlgebra
import ..FermionExperiment as FE
export real_correlation_matrix, entropy2_real!, entropy_rectangle_real

"""
Reconstruct a Float64 covariance directly, without allocating a dense complex
matrix. Returns (C, checks). The imaginary part and Hermitian error are measured
before discarding them. For a physical covariance, the second-order entropy
change from taking its real part is bounded by 4*norm(imag(C))^2 (Frobenius).
"""
function real_correlation_matrix(state::FE.FermionState, indices;
        imag_atol=2e-10, entropy_error_atol=1e-9, projector_atol=1e-9)
    projector_error=maximum(norm(q'*q-I) for q in state.orbitals)
    projector_error <= projector_atol || error("orbitals are not orthonormal: $projector_error")
    model=state.prepared.model
    kernel=FE.correlation_kernel(state)
    coords=[FE.site_coordinates(model,i) for i in indices]
    n=length(indices)
    C=Matrix{Float64}(undef,n,n)
    maximag=0.0; imag_norm2=0.0; maxhermitian=0.0
    @inbounds for j in 1:n, i in j:n
        a,x,y=coords[i]; b,u,v=coords[j]
        z=kernel[a,b,mod(x-u,model.q)+1,mod(y-v,model.q)+1]
        zt=conj(kernel[b,a,mod(u-x,model.q)+1,mod(v-y,model.q)+1])
        value=(z+zt)/2
        C[i,j]=C[j,i]=real(value)
        maximag=max(maximag,abs(imag(value)))
        imag_norm2 += (i==j ? 1 : 2)*abs2(imag(value))
        maxhermitian=max(maxhermitian,abs(z-zt))
    end
    bound=4imag_norm2
    maximag <= imag_atol || error("covariance is not sufficiently real: $maximag")
    bound <= entropy_error_atol || error("real-part entropy error bound too large: $bound")
    checks=(n=n,max_imaginary=maximag,imaginary_frobenius=sqrt(imag_norm2),
        real_part_entropy_bound=bound,kernel_hermitian_error=maxhermitian,
        orbital_projector_error=projector_error,matrix_bytes=sizeof(C))
    return C,checks
end

"""
Compute single-species S2 and DESTROY C. :eigen checks the covariance spectrum.
:cholesky requires a covariance certified by construction. It uses
  C^2+(I-C)^2 = 0.5I + 2(C-0.5I)^2,
whose eigenvalues lie in [0.5,1], so its condition number is at most two.
The SYRK and Cholesky path has no cancellation in forming the positive matrix.
"""
function entropy2_real!(C::Matrix{Float64}; method=:cholesky, copies=1,
        covariance_validated=false)
    size(C,1)==size(C,2) || throw(DimensionMismatch("covariance must be square"))
    if method==:eigen
        values=eigvals!(Symmetric(C,:L))
        minimum(values)>=-2e-9 && maximum(values)<=1+2e-9 ||
            error("correlation eigenvalues outside [0,1]")
        values=clamp.(values,0.0,1.0)
        return -copies*sum(log1p.(-2 .* values .* (1 .- values)))
    elseif method==:cholesky
        covariance_validated || error("Cholesky entropy requires a covariance certified by construction")
        n=size(C,1)
        @inbounds for i in 1:n
            C[i,i]-=0.5
        end
        A=Matrix{Float64}(undef,n,n)
        BLAS.syrk!('L','N',2.0,C,0.0,A)
        @inbounds for i in 1:n
            A[i,i]+=0.5
        end
        factor=cholesky!(Symmetric(A,:L))
        entropy=-2copies*sum(log(factor.factors[i,i]) for i in 1:n)
        entropy>=-1e-8 || error("negative entropy from purported covariance: $entropy")
        return entropy
    end
    throw(ArgumentError("method must be :eigen or :cholesky"))
end

"""
Drop-in rectangle entropy with default Cholesky and optional numerical checks.
Full-span strips retain FE's Fourier decomposition. with_checks=true returns
(entropy, checks); otherwise it returns only entropy, matching FE's API.
"""
function entropy_rectangle_real(state::FE.FermionState; width=state.prepared.model.L÷3,
        height=2state.prepared.model.L÷3,x0=1,y0=1,copies=1,method=:cholesky,
        with_checks=false,imag_atol=2e-10,entropy_error_atol=1e-9)
    L=state.prepared.model.L
    if (height==L && y0==1) || (width==L && x0==1)
        entropy=FE.entropy_rectangle(state;width,height,x0,y0,copies)
        return with_checks ? (entropy,(method=:strip,)) : entropy
    end
    indices=FE.rectangle_indices(L;width,height,x0,y0)
    C,checks=real_correlation_matrix(state,indices;imag_atol,entropy_error_atol)
    entropy=entropy2_real!(C;method,copies,covariance_validated=true)
    return with_checks ? (entropy,merge(checks,(;method))) : entropy
end
end
