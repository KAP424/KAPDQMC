#= Independent small-system audit. Never executes Free_M.jl's default scan.

Run: julia --project=. src/Measure/born_ai/analysis/audit_free_m_updates.jl
The historical implementation is loaded only up to its top-level L_set assignment.
References rebuild the weighted-orbital Gram matrix by Cholesky/logdet and the
site-space inverse separately; no entropy or large-L production jobs are run.
=#
using LinearAlgebra, Random, SHA, TOML, Dates
BLAS.set_num_threads(1)
const HERE = @__DIR__
const SNAPSHOT = joinpath(HERE, "Free_M_audit_snapshot.jl")
const ORIGINAL_SHA = "5bdafd14523deaeb31034f39ef55ab279e327ad6320866e310028d35023750e0"
@assert bytes2hex(sha256(read(SNAPSHOT))) == ORIGINAL_SHA
const TEXT = first(split(read(SNAPSHOT, String), "\nL_set ="))
@assert !occursin("for L in L_set", TEXT)
include_string(Main, TEXT, SNAPSHOT)

# Add diagnostics without changing arithmetic, random calls, or acceptance rules.
const COUNTS = Dict("proposals"=>0, "accepted"=>0, "negative"=>0, "nonfinite"=>0)
counted_text = replace(TEXT,
    "function UpdateMeasure!"=>"function CountedUpdateMeasure!",
    "ratio = Δ * M.X[j, j]"=>"ratio = Δ * M.X[j, j]\n        COUNTS[\"proposals\"] += 1\n        COUNTS[\"negative\"] += ratio < 0\n        COUNTS[\"nonfinite\"] += !isfinite(ratio)",
    "if rand() < ratio"=>"if rand() < ratio\n            COUNTS[\"accepted\"] += 1")
counted_start = findfirst("function CountedUpdateMeasure!", counted_text).start
counted_stop = findfirst("struct tmpME_", counted_text).start - 1
include_string(Main, counted_text[counted_start:counted_stop], "counted_snapshot")

function setup(L)
    K = nnK_Matrix("HoneyComb120", [L,L])
    E,V = LAPACK.syevd!('V','L',K)
    n = L^2
    P = V[:,1:n]
    # Match the historical implementation, including its explicit Gram inverse.
    P, P*inv(P'*P)*P'
end

function high_precision_ratio(row,seed)
    L=row["L"]; beta=row["beta"]; init=row["initial"]
    P,G=setup(L); N,n=size(P)
    rng=MersenneTwister(seed)
    idx=init=="random" ? randperm(rng,N)[1:n] : init=="A" ? collect(1:2:N) : collect(1:n)
    comp=setdiff(1:N,idx); i,j=row["worst_proposal"]
    idx2=copy(idx); idx2[j]=comp[i]
    A,_=ref_inverse(G,idx,beta)
    r,_=single_update(inv(A),G,idx,comp,i,j,beta)
    r64=exp(ref_logweight(P,idx2,beta)-ref_logweight(P,idx,beta))
    rbig=setprecision(256) do
        Q=BigFloat.(P)
        function lw(ii)
            w=ones(BigFloat,N); w[ii].=exp(-2BigFloat(beta))
            Y=w.*Q
            F=cholesky(Symmetric(Y'*Y))
            2sum(log,diag(F.U))
        end
        exp(lw(idx2)-lw(idx))
    end
    Dict("L"=>L,"beta"=>beta,"initial"=>init,"proposal"=>[i,j],"precision_bits"=>256,
      "reference_ratio"=>string(rbig),"source_ratio"=>r,"cholesky_float64_ratio"=>r64,
      "source_relative_error"=>Float64(abs(BigFloat(r)-rbig)/rbig),
      "cholesky_float64_relative_error"=>Float64(abs(BigFloat(r64)-rbig)/rbig))
end
function ref_logweight(P, idx, beta)
    v = ones(size(P,1)); v[idx] .= exp(-2beta)
    Y = v .* P
    F = cholesky(Symmetric(Y'*Y))
    2sum(log, diag(F.U))
end
function ref_inverse(G, idx, beta)
    A = Matrix{Float64}(I,length(idx),length(idx)) - (1-exp(-4beta))*G[idx,idx]
    F = cholesky(Symmetric(A))
    A, inv(F)
end
function single_update(X,G,idx,comp,i,j,beta)
    lambda = 1-exp(-4beta)
    g = G[idx,comp[i]]
    xg = X*g
    z = xg - (xg[j]/X[j,j])*X[:,j]
    delta = 1-lambda*G[comp[i],comp[i]]-lambda^2*dot(G[comp[i],idx],z)
    r = delta*X[j,j]
    Y = X-X[:,j]*X[j,:]'/X[j,j]+lambda^2/delta*(z*z')
    Y[:,j] .= lambda/delta .* z
    Y[j,:] .= lambda/delta .* z
    Y[j,j] = 1/delta
    r,Y
end

function proposal_audit(L,beta,init,seed)
    rng=MersenneTwister(seed)
    P,G=setup(L); N,n=size(P)
    idx = init=="random" ? randperm(rng,N)[1:n] : init=="A" ? collect(1:2:N) : collect(1:n)
    comp=setdiff(1:N,idx)
    A,_=ref_inverse(G,idx,beta); X=inv(A); logold=ref_logweight(P,idx,beta)
    maxre=0.; maxxe=0.; maxres=0.; maxlogerr=0.; negative=0
    worst=Int[]
    for i in 1:n,j in 1:n
        idx2=copy(idx); idx2[j]=comp[i]
        true_ratio=exp(ref_logweight(P,idx2,beta)-logold)
        r,Y=single_update(X,G,idx,comp,i,j,beta)
        A2,X2=ref_inverse(G,idx2,beta)
        re=abs(r-true_ratio)/true_ratio
        if re>maxre; maxre=re; worst=[i,j]; end
        maxxe=max(maxxe,norm(Y-X2)/norm(X2))
        maxres=max(maxres,norm(A2*Y-I,Inf))
        maxlogerr=max(maxlogerr,abs(log(max(r,floatmin()))-log(true_ratio)))
        negative += r<=0
    end
    Dict("L"=>L,"beta"=>beta,"initial"=>init,"proposals"=>n^2,
      "lambda"=>1-exp(-4beta),"A_condition"=>cond(A),"worst_proposal"=>worst,
      "max_ratio_relative_error"=>maxre,"max_logratio_error"=>maxlogerr,
      "max_inverse_relative_error"=>maxxe,"max_inverse_residual_inf"=>maxres,"nonpositive_ratio"=>negative)
end

function drift_audit(L,beta,init,refresh,sweeps,seed)
    Random.seed!(seed)
    P,G=setup(L); N,n=size(P)
    M=Measure_EE(beta,N,n)
    if init=="A"; M.E_idx=collect(1:2:N); M.S_idx=setdiff(1:N,M.E_idx); end
    if init=="contiguous"; M.E_idx=collect(1:n); M.S_idx=setdiff(1:N,M.E_idx); end
    A,_=ref_inverse(G,M.E_idx,beta); M.X=inv(A)
    v=zeros(n); vv=zeros(n,n)
    # Prove the instrumented method produces exactly the same trajectory as source.
    Q=deepcopy(M); R=deepcopy(M)
    Random.seed!(seed+1); UpdateMeasure!(Q,G,v,vv,P)
    Random.seed!(seed+1); CountedUpdateMeasure!(R,G,v,vv,P)
    @assert Q.E_idx==R.E_idx && Q.X==R.X && Q.MO==R.MO
    for k in keys(COUNTS); COUNTS[k]=0; end
    Random.seed!(seed+2)
    maxrel=0.; maxres=0.; maxasym=0.; worst=0; failed=""
    checkpoints=Dict{String,Any}[]
    started=time()
    for sweep in 1:sweeps
        # Matches the original: refresh BEFORE sweep 100, 200, ... .
        if refresh>0 && sweep%refresh==0
            M.X=inv(Matrix{Float64}(I,n,n)-M.λ*G[M.E_idx,M.E_idx])
        end
        CountedUpdateMeasure!(M,G,v,vv,P)
        if sweep%10==0 || sweep==1 || sweep%100==99
            A,X=ref_inverse(G,M.E_idx,beta)
            rel=norm(M.X-X)/norm(X); res=norm(A*M.X-I,Inf)
            asym=norm(M.X-M.X',Inf)
            if rel>maxrel; maxrel=rel; worst=sweep; end
            maxres=max(maxres,res); maxasym=max(maxasym,asym)
            push!(checkpoints,Dict("sweep"=>sweep,"inverse_relative_error"=>rel,"inverse_residual_inf"=>res,"condition"=>cond(A)))
            if !isfinite(rel) || rel>1e-3
                failed="stopped when inverse relative error exceeded 1e-3"
                break
            end
        end
    end
    Dict("L"=>L,"beta"=>beta,"initial"=>init,"refresh_every_sweeps"=>refresh,"requested_sweeps"=>sweeps,
      "proposals"=>COUNTS["proposals"],"accepted"=>COUNTS["accepted"],"acceptance"=>COUNTS["accepted"]/COUNTS["proposals"],
      "negative_ratio"=>COUNTS["negative"],"nonfinite_ratio"=>COUNTS["nonfinite"],"max_inverse_relative_error"=>maxrel,
      "max_inverse_residual_inf"=>maxres,"max_asymmetry_inf"=>maxasym,"worst_sweep"=>worst,
      "stop_reason"=>failed,"checkpoints"=>checkpoints,"elapsed_seconds"=>time()-started)
end

function purity_audit(L,beta)
    Random.seed!(458+L)
    P,G=setup(L); N,n=size(P)
    M=Measure_EE(beta,N,n); M.MO[M.E_idx].=exp(-2beta)
    M.M_BR.=M.MO.*P; tau=zeros(n)
    LAPACK.geqrf!(M.M_BR,tau); LAPACK.orgqr!(M.M_BR,tau,n)
    M.M_BL.=P'.*M.MO'
    LAPACK.gerqf!(M.M_BL,tau); LAPACK.orgrq!(M.M_BL,tau,n)
    KAPDQMC.tUDQMC.get_G!(zeros(n,n),zeros(n,N),zeros(LAPACK.BlasInt,n),M.M_BL,M.M_BR,M.M_G)
    Y=M.MO.*P; Q=Matrix(qr(Y).Q)[:,1:n]; C=Q*Q'
    maxerr=0.; vals=Dict{String,Any}[]
    for (name,extent) in [("A",[L÷3,L]),("B",[L÷3,2L÷3])]
        idx=area_index("HoneyComb120",[L,L],([1,1],extent))
        GM=zeros(length(idx),length(idx)); H=M.M_G[idx,idx]
        GroverMatrix!(GM,H,H)
        nu=eigvals(Symmetric(C[idx,idx]))
        s2=-sum(log.(nu.^2+(1 .-nu).^2))
        source_s2=-log(det(GM)); maxerr=max(maxerr,abs(s2-source_s2))
        push!(vals,Dict("region"=>name,"source_purity"=>det(GM),"source_S2"=>source_s2,"direct_S2"=>s2))
    end
    Dict("L"=>L,"beta"=>beta,"hole_C_error_inf"=>norm(M.M_G-(I-C),Inf),"max_S2_error"=>maxerr,"regions"=>vals)
end

function main()
    started=time()
    rows=Dict{String,Any}[]; drift=Dict{String,Any}[]; purity=Dict{String,Any}[]
    for L in (3,6), beta in (0.,.5,1.,2.,3.,5.), initial in ("random","A","contiguous")
        push!(rows,proposal_audit(L,beta,initial,1551+L))
    end
    for L in (3,6), beta in (0.,.5,1.,2.,3.,5.), initial in ("random","A")
        row=drift_audit(L,beta,initial,100,1000,9481+L)
        push!(drift,row)
        println("L=$L beta=$beta initial=$initial acceptance=$(row["acceptance"]) drift=$(row["max_inverse_relative_error"])")
        flush(stdout)
    end
    for (L,beta,initial,sweeps) in [(12,3.,"random",1000),(12,5.,"random",1000),
                                   (12,3.,"contiguous",300),(12,5.,"contiguous",300),
                                   (18,3.,"random",300),(18,5.,"random",300)]
        row=drift_audit(L,beta,initial,100,sweeps,9481+L)
        push!(drift,row)
        println("STRESS L=$L beta=$beta initial=$initial acceptance=$(row["acceptance"]) drift=$(row["max_inverse_relative_error"]) elapsed=$(row["elapsed_seconds"])s")
        flush(stdout)
    end
    for L in (3,6), beta in (0.,.5,1.,2.,3.,5.)
        push!(purity,purity_audit(L,beta))
    end
    hp=[high_precision_ratio(r,1551+r["L"]) for r in rows if r["beta"]>=2]
    out=Dict("created_at"=>string(now()),"snapshot_sha256"=>ORIGINAL_SHA,"snapshot"=>SNAPSHOT,
      "audit_script_sha256"=>bytes2hex(sha256(read(@__FILE__))),"julia_version"=>string(VERSION),
      "elapsed_seconds"=>time()-started,"blas_threads"=>BLAS.get_num_threads(),
      "proposal_tests"=>rows,"drift_tests"=>drift,"purity_tests"=>purity,"high_precision_ratio_tests"=>hp)
    open(joinpath(HERE,"free_m_update_audit.toml"),"w") do io; TOML.print(io,out); end
    println("audit complete elapsed=$(out["elapsed_seconds"])s")
end
if abspath(PROGRAM_FILE)==@__FILE__; main(); end
