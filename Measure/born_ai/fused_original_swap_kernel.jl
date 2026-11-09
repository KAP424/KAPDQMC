module FusedOriginalSwapKernel
using LinearAlgebra, Random
export sweep_fused!

"""Uniform E/S single-swap Metropolis; fuse only the two accepted inverse rank updates."""
function sweep_fused!(state,G,rng=Random.default_rng())
    n=length(state.E); lambda=state.lambda; accepted=0
    for attempt in 1:2n
        i=rand(rng,1:n); j=rand(rng,1:n); s=state.S[i]
        @inbounds for k in 1:n
            state.g[k]=G[state.E[k],s]
            state.grow[k]=G[s,state.E[k]]
        end
        mul!(state.xg,state.X,state.g)
        xjj=state.X[j,j]; f=state.xg[j]/xjj
        @inbounds for k in 1:n; state.z[k]=state.xg[k]-f*state.X[k,j]; end
        state.z[j]=0.
        delta=1-lambda*G[s,s]-lambda^2*dot(state.grow,state.z)
        ratio=delta*xjj
        isfinite(ratio) && ratio>0 && delta>0 && xjj>0 || error("invalid original-swap ratio")
        if rand(rng)<min(1.,ratio)
            @inbounds for k in 1:n
                state.column[k]=state.X[k,j]; state.row[k]=state.X[j,k]
            end
            # Cached pre-update vectors; one pass over X and no fastmath.
            alpha=-1/xjj; gamma=lambda^2/delta
            @inbounds for k in 1:n
                row_k=state.row[k]; z_k=state.z[k]
                @simd for ell in 1:n
                    state.X[ell,k]+=alpha*state.column[ell]*row_k+gamma*state.z[ell]*z_k
                end
            end
            f=lambda/delta
            @inbounds for k in 1:n
                state.X[k,j]=f*state.z[k]; state.X[j,k]=f*state.z[k]
            end
            state.X[j,j]=1/delta
            state.E[j],state.S[i]=state.S[i],state.E[j]
            state.log_weight+=log(ratio)
            accepted+=1
        end
    end
    (;accepted,proposed=2n)
end
end # module
