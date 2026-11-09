"""Optional exact half-size rectangle entropy for the uniform-A protocol.

The existing evolution and entropy kernels are not changed. Applicability is
checked explicitly: one A-occupied orbital per primitive cell, real covariance,
complementary diagonal sublattice blocks, and the rectangle inversion symmetries.
"""
module UniformAEntropyExtension
using LinearAlgebra
import ..FermionExperiment as FE
export build_half_matrix, entropy_rectangle_half

"""
Return M=Z+iY for the cell-ordered rectangle, with Z=C_AA-I/2 and Y=C_AB*J.
J reverses the rectangular cell order, i.e. simultaneously inverts x and y.
Only one half-size complex matrix is allocated; no full rectangle covariance.
"""
function build_half_matrix(state::FE.FermionState;
        width=state.prepared.model.L÷3,height=2state.prepared.model.L÷3,
        x0=1,y0=1,imag_atol=2e-10,structure_atol=1e-10,
        entropy_error_atol=1e-9,projector_atol=1e-10)
    pre=state.prepared
    model=pre.model
    model.m==1 && pre.F==[1] && pre.E==[2] ||
        throw(ArgumentError("half-size extension requires the m=1 uniform_A initial pattern"))
    L=model.L
    1<=x0<=x0+width-1<=L && 1<=y0<=y0+height-1<=L ||
        throw(ArgumentError("rectangle must lie within the lattice"))
    projector_error=maximum(norm(q'*q-I) for q in state.orbitals)
    projector_error<=projector_atol || error("orbitals are not orthonormal: $projector_error")
    kernel=FE.correlation_kernel(state)
    cells=[(x,y) for y in y0-1:y0+height-2 for x in x0-1:x0+width-2]
    p=length(cells)
    M=Matrix{ComplexF64}(undef,p,p)
    maximag=0.0; imag_norm2=0.0; complement_error=0.0; hermitian_error=0.0
    @inbounds for j in 1:p, i in 1:p
        x,y=cells[i]; u,v=cells[j]
        dx,dy=mod(x-u,L)+1,mod(y-v,L)+1
        rx,ry=mod(u-x,L)+1,mod(v-y,L)+1
        aa,ab=kernel[1,1,dx,dy],kernel[1,2,dx,dy]
        ba,bb=kernel[2,1,dx,dy],kernel[2,2,dx,dy]
        maximag=max(maximag,abs(imag(aa)),abs(imag(ab)),abs(imag(ba)),abs(imag(bb)))
        imag_norm2+=abs2(imag(aa))+abs2(imag(ab))+abs2(imag(ba))+abs2(imag(bb))
        complement_error=max(complement_error,abs(aa+bb-(i==j ? 1 : 0)))
        hermitian_error=max(hermitian_error,abs(aa-conj(kernel[1,1,rx,ry])),
            abs(bb-conj(kernel[2,2,rx,ry])),abs(ab-conj(kernel[2,1,rx,ry])))
        ju,jv=cells[p+1-j]
        yij=kernel[1,2,mod(x-ju,L)+1,mod(y-jv,L)+1]
        M[i,j]=complex(real(aa)-(i==j ? .5 : 0),real(yij))
    end
    zsym=0.0; ysym=0.0; zcentro=0.0
    @inbounds for j in 1:p, i in 1:p
        zsym=max(zsym,abs(real(M[i,j])-real(M[j,i])))
        ysym=max(ysym,abs(imag(M[i,j])-imag(M[j,i])))
        zcentro=max(zcentro,abs(real(M[i,j])-real(M[p+1-i,p+1-j])))
    end
    structure=max(complement_error,hermitian_error,zsym,ysym,zcentro)
    real_bound=4imag_norm2
    maximag<=imag_atol || error("covariance is not real to tolerance: $maximag")
    real_bound<=entropy_error_atol || error("real-part entropy bound too large: $real_bound")
    structure<=structure_atol || error("half-size inversion/block identity failed: $structure")
    # Remove only the roundoff antisymmetric part after checking it explicitly.
    @inbounds for j in 1:p, i in j+1:p
        value=(M[i,j]+M[j,i])/2
        M[i,j]=M[j,i]=value
    end
    checks=(n=2p,half_n=p,max_imaginary=maximag,imaginary_frobenius=sqrt(imag_norm2),
        real_part_entropy_bound=real_bound,complement_error=complement_error,
        kernel_hermitian_error=hermitian_error,z_symmetric_error=zsym,
        y_symmetric_error=ysym,z_centrosymmetric_error=zcentro,max_structure_error=structure,
        orbital_projector_error=projector_error,half_matrix_bytes=sizeof(M))
    return M,checks
end

"""
With K=C-I/2 transformed to [Z Y;Y -Z], M=Z+iY is complex symmetric.
K has eigenvalues ±svd(M), so S2=-2*logdet(0.5I+2M†M). HERK plus
Cholesky operates on half the full covariance dimension. The eigenvalues
of its positive matrix lie in [0.5,1] for a valid covariance.
"""
function entropy_rectangle_half(state::FE.FermionState;
        width=state.prepared.model.L÷3,height=2state.prepared.model.L÷3,
        x0=1,y0=1,copies=1,with_checks=false,kwargs...)
    M,checks=build_half_matrix(state;width,height,x0,y0,kwargs...)
    p=size(M,1)
    positive=Matrix{ComplexF64}(undef,p,p)
    BLAS.herk!('L','C',2.0,M,0.0,positive)
    @inbounds for i in 1:p
        positive[i,i]+=.5
    end
    factor=cholesky!(Hermitian(positive,:L))
    entropy=-4copies*sum(log(real(factor.factors[i,i])) for i in 1:p)
    entropy>=-1e-8 || error("negative entropy from half-size construction: $entropy")
    result=merge(checks,(method=:uniformA_half_cholesky,workspace_matrix_bytes=sizeof(M)+sizeof(positive)))
    return with_checks ? (entropy,result) : entropy
end
end
