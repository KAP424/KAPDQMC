"""Optional full-spectrum shifted-determinant Renyi-2 entropy for real covariances.

No evolution, measurement, or existing entropy kernel is changed. This module
uses neither sublattice complementarity nor the uniform-A half-size identity.
"""
module MultiPatternEntropyExtension
using LinearAlgebra
import ..FermionExperiment as FE
export shifted_correlation_matrix, entropy2_shift!, entropy_rectangle_shift

"""Construct A=C-(1+i)I/2 directly as one full-size complex symmetric matrix.

As in the real-covariance extension, reality and orbital normalization are checked
before discarding roundoff imaginary parts. Every spatial degree of freedom and
every mode of the supplied state is retained.
"""
function shifted_correlation_matrix(state::FE.FermionState, indices;
        imag_atol=2e-10,entropy_error_atol=1e-9,projector_atol=1e-9,
        hermitian_atol=1e-9)
    projector_error=maximum(norm(q'*q-I) for q in state.orbitals)
    projector_error<=projector_atol || error("orbitals are not orthonormal: $projector_error")
    model=state.prepared.model
    kernel=FE.correlation_kernel(state)
    coords=[FE.site_coordinates(model,i) for i in indices]
    n=length(indices)
    A=Matrix{ComplexF64}(undef,n,n)
    maximag=0.0;imag_norm2=0.0;maxhermitian=0.0
    @inbounds for j in 1:n,i in j:n
        a,x,y=coords[i];b,u,v=coords[j]
        z=kernel[a,b,mod(x-u,model.q)+1,mod(y-v,model.q)+1]
        zt=conj(kernel[b,a,mod(u-x,model.q)+1,mod(v-y,model.q)+1])
        value=(z+zt)/2
        A[i,j]=A[j,i]=complex(real(value),0.0)
        maximag=max(maximag,abs(imag(value)))
        imag_norm2+=(i==j ? 1 : 2)*abs2(imag(value))
        maxhermitian=max(maxhermitian,abs(z-zt))
    end
    @inbounds for i in 1:n
        A[i,i]-=complex(.5,.5)
    end
    bound=4imag_norm2
    maximag<=imag_atol || error("covariance is not sufficiently real: $maximag")
    bound<=entropy_error_atol || error("real-part entropy error bound too large: $bound")
    maxhermitian<=hermitian_atol || error("covariance Hermitian error: $maxhermitian")
    checks=(n=n,max_imaginary=maximag,imaginary_frobenius=sqrt(imag_norm2),
        real_part_entropy_bound=bound,kernel_hermitian_error=maxhermitian,
        orbital_projector_error=projector_error,workspace_matrix_bytes=sizeof(A))
    return A,checks
end

"""Compute S2=-n*log(2)-2*logabsdet(A), DESTROYING shifted A.

For a real symmetric physical covariance, C^2+(I-C)^2=2AA* and the
singular values of A lie in [1/2,1/sqrt(2)]. logabsdet avoids any choice
of complex-log branch. The dimension is the FULL spatial covariance size.
"""
function entropy2_shift!(A::Matrix{ComplexF64};method=:bunchkaufman,copies=1,
        covariance_validated=false,with_checks=false)
    n=size(A,1)
    size(A,2)==n || throw(DimensionMismatch("shifted covariance must be square"))
    covariance_validated || error("shift determinant requires a covariance certified by construction")
    if method==:bunchkaufman
        factor=bunchkaufman!(Symmetric(A,:L);check=true)
    elseif method==:bunchkaufman_rook
        factor=bunchkaufman!(Symmetric(A,:L),true;check=true)
    elseif method==:lu
        factor=lu!(A;check=true)
    else
        throw(ArgumentError("method must be :bunchkaufman, :bunchkaufman_rook or :lu"))
    end
    logarithm,phase=logabsdet(factor)
    isfinite(logarithm) || error("nonfinite shifted determinant")
    entropy=copies*(-n*log(2.0)-2logarithm)
    entropy>=-1e-8 || error("negative entropy from shifted determinant: $entropy")
    checks=(method=method,log_determinant_abs=logarithm,
        determinant_phase_modulus_error=abs(abs(phase)-1),workspace_matrix_bytes=sizeof(A))
    return with_checks ? (entropy,checks) : entropy
end

"""Drop-in optional rectangle entropy; full-span strips keep FE's Fourier path."""
function entropy_rectangle_shift(state::FE.FermionState;
        width=state.prepared.model.L÷3,height=2state.prepared.model.L÷3,x0=1,y0=1,
        copies=1,method=:bunchkaufman,with_checks=false,kwargs...)
    L=state.prepared.model.L
    if (height==L && y0==1)||(width==L && x0==1)
        entropy=FE.entropy_rectangle(state;width,height,x0,y0,copies)
        return with_checks ? (entropy,(method=:strip,)) : entropy
    end
    indices=FE.rectangle_indices(L;width,height,x0,y0)
    A,checks=shifted_correlation_matrix(state,indices;kwargs...)
    entropy,factor_checks=entropy2_shift!(A;method,copies,covariance_validated=true,with_checks=true)
    return with_checks ? (entropy,merge(checks,factor_checks)) : entropy
end
end
