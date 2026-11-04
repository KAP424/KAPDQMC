# Same paired observables and original swap as the frozen diagnostic runner.
# One symmetric full correlation matrix replaces repeated regional V*V'.
# The untyped original entropy method remains callable with invoke for audits.
include("run_born_boundary_check.jl")

const BOUNDARY_FULL_CORRELATION = Ref{Matrix{Float64}}(zeros(0,0))

function boundary_entropies!(work::BoundaryWorkspace,model::GroundModel,E,beta)
    work.Q.=model.Q
    attenuation=exp(-2beta)
    @inbounds for j in 1:model.n,i in E;work.Q[i,j]*=attenuation;end
    LAPACK.geqrf!(work.Q,work.tau)
    LAPACK.orgqr!(work.Q,work.tau,model.n)
    if size(BOUNDARY_FULL_CORRELATION[])!=(model.N,model.N)
        BOUNDARY_FULL_CORRELATION[]=zeros(model.N,model.N)
    end
    G=BOUNDARY_FULL_CORRELATION[]
    BLAS.syrk!('U','N',1.,work.Q,0.,G)
    @inbounds for j in 1:model.N,i in j+1:model.N;G[i,j]=G[j,i];end
    values=zeros(length(work.indices))
    for (ri,idx) in enumerate(work.indices)
        k=length(idx);C=view(work.C,1:k,1:k);B=view(work.B,1:k,1:k)
        @inbounds for j in 1:k,i in 1:k;C[i,j]=G[idx[i],idx[j]];end
        @inbounds for i in 1:k;C[i,i]-=.5;end
        BLAS.syrk!('U','N',2.,C,0.,B)
        @inbounds for i in 1:k;B[i,i]+=.5;end
        factor=cholesky!(Symmetric(B,:U))
        values[ri]=-2sum(log,diag(factor.U))
    end
    map(work.entries) do entry
        SA,SB,SC=values[collect(entry.ids)]
        D=((entry.m-1)*SA+SC)/entry.m-SB
        (;entry...,SA,SB,SC,D)
    end
end

if abspath(PROGRAM_FILE)==@__FILE__;boundary_main();end
