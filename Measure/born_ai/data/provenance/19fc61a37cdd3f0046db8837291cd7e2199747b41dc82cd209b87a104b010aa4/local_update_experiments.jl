# Sampling-only audit. Never includes Free_M.jl or performs entropy calculations.
push!(LOAD_PATH, normpath(joinpath(@__DIR__, "..", "..")))
using KAPDQMC, LinearAlgebra, Random, Statistics, SHA, TOML, Printf

const ROOT=@__DIR__
const DATA=joinpath(ROOT,"data")
mkpath(DATA)
BLAS.set_num_threads(1)
arg(name,default)=let xs=filter(x->startswith(x,"--$name="),ARGS); isempty(xs) ? default : split(last(xs),'=';limit=2)[2] end
sha(path)=bytes2hex(sha256(read(path)))

function model(L)
    K=nnK_Matrix("HoneyComb120",[L,L]); energies,V=LAPACK.syevd!('V','L',K)
    a=L^2; P=copy(V[:,1:a]); G=P*P'
    open(joinpath(DATA,"local_projector_L$(L).bin"),"w") do io;write(io,G);end
    open(joinpath(DATA,"local_orbitals_L$(L).bin"),"w") do io;write(io,P);end
    (;L,a,N=2a,P,G,energies)
end

function initial(L,kind,rng)
    N=2L^2
    kind=="uniform_A" && return collect(1:2:N)
    kind=="random" && return sort(randperm(rng,N)[1:L^2])
    result=Int[]
    for y in 1:L,x in 1:L
        A=2(x+(y-1)*L)-1;B=A+1
        if kind=="AB_checker"
            push!(result,iseven(x+y) ? A : B)
        elseif kind=="02_stripe"
            isodd(x) && append!(result,(A,B))
        elseif kind=="02_checker"
            iseven(x+y) && append!(result,(A,B))
        elseif kind=="mixed"
            cell=1+(x-1)%2+2*((y-1)%2)
            cell==1 ? append!(result,(A,B)) : cell==2 ? push!(result,A) : cell==3 ? push!(result,B) : nothing
        else
            error("Unknown initial state")
        end
    end
    @assert length(result)==L^2
    result
end

function logweight(G,E,beta)
    beta==0 && return 0.
    λ=-expm1(-4beta)
    B=Matrix{Float64}(I,length(E),length(E))-λ*G[E,E]
    F=cholesky(Symmetric(B))
    2sum(log,diag(F.L))
end

function normalizer(a,beta)
    values=zeros(a+1); lc=0.
    for j in 0:a
        j>0 && (lc+=log(a-j+1)-log(j))
        values[j+1]=2lc-4beta*j
    end
    high=maximum(values);high+log(sum(exp.(values.-high)))
end

function metrics(E,N,start)
    occupied=falses(N);occupied[E].=true;a=N÷2
    imbalance=(sum(occupied[1:2:N])-sum(occupied[2:2:N]))/a
    n02=count(i->occupied[2i-1]==occupied[2i],1:a)/a
    memory=(2sum(occupied[start])/a)-1
    imbalance,n02,memory
end

mutable struct Chain
    E::Vector{Int};S::Vector{Int};X::Matrix{Float64};g::Vector{Float64}
    tmp::Vector{Float64};z::Vector{Float64};col::Vector{Float64};logw::Float64
end
function rebuild!(chain,G,beta)
    a=length(chain.E);λ=-expm1(-4beta)
    B=Matrix{Float64}(I,a,a)-λ*G[chain.E,chain.E]
    F=cholesky(Symmetric(B));truth=Matrix(inv(F));value=2sum(log,diag(F.L))
    drift=norm(chain.X-truth)/max(norm(truth),eps());logdrift=abs(chain.logw-value)
    chain.X.=truth;chain.logw=value
    drift,logdrift
end
function chain(G,E,beta)
    a=length(E);N=size(G,1)
    c=Chain(copy(E),setdiff(1:N,E),Matrix{Float64}(I,a,a),zeros(a),zeros(a),zeros(a),zeros(a),0.)
    rebuild!(c,G,beta);c
end

function sweep!(c,G,beta,rng)
    a=length(c.E);λ=-expm1(-4beta);accepted=0
    for step in 1:2a
        i=rand(rng,1:a);j=rand(rng,1:a)
        if beta==0
            c.E[j],c.S[i]=c.S[i],c.E[j];accepted+=1;continue
        end
        s=c.S[i]
        @inbounds for k in 1:a;c.g[k]=G[c.E[k],s];end
        mul!(c.tmp,c.X,c.g);xjj=c.X[j,j];factor=c.tmp[j]/xjj
        @inbounds for k in 1:a;c.z[k]=c.tmp[k]-factor*c.X[k,j];end
        c.z[j]=0.
        delta=1-λ*G[s,s]-λ^2*dot(c.g,c.z);ratio=delta*xjj
        isfinite(ratio)&&ratio>0 || error("Nonpositive/invalid ratio: beta=$beta ratio=$ratio")
        if rand(rng)<ratio
            @inbounds for k in 1:a;c.col[k]=c.X[k,j];end
            BLAS.ger!(-1/xjj,c.col,c.col,c.X)
            BLAS.ger!(λ^2/delta,c.z,c.z,c.X)
            factor=λ/delta
            @inbounds for k in 1:a;c.X[k,j]=factor*c.z[k];c.X[j,k]=factor*c.z[k];end
            c.X[j,j]=1/delta
            c.E[j],c.S[i]=c.S[i],c.E[j];c.logw+=log(ratio);accepted+=1
        end
    end
    accepted/(2a)
end

function exact_small(m)
    a=m.a;N=m.N;sets=[findall(i->(mask>>(i-1))&1==1,1:N) for mask in 0:(1<<N)-1 if count_ones(mask)==a]
    keys=Dict(sum(1<<(i-1) for i in E)=>j for (j,E) in enumerate(sets))
    open(joinpath(DATA,"local_exact_probabilities.csv"),"w") do points
        open(joinpath(DATA,"local_exact_summary.csv"),"w") do summary
            println(points,"L,beta,state,occupied_bits,class_id,imbalance,n02_fraction,log_weight,probability")
            println(summary,"L,beta,states,logZ_enumerated,logZ_formula,normalization_error,mean_acceptance,gap_single_proposal,absolute_gap_single_proposal,gap_one_sweep,p_class1,p_class2,p_class3,p_uniform_A,p_AB_checker,p_02_stripe,p_02_checker,p_mixed")
            for beta in (0.,.25,.5,1.,2.,3.)
                lw=[logweight(m.G,E,beta) for E in sets];z=maximum(lw)+log(sum(exp.(lw.-maximum(lw))));p=exp.(lw.-z)
                T=zeros(length(sets),length(sets));classes=zeros(3)
                for (row,E) in enumerate(sets)
                    S=setdiff(1:N,E);imb,n02,_=metrics(E,N,E);cls=n02==0 ? 1 : n02==1 ? 2 : 3
                    classes[cls]+=p[row]
                    bits=join(Int(i in E) for i in 1:N)
                    println(points,join((m.L,beta,row,bits,cls,imb,n02,lw[row],p[row]),','))
                    for e in E,s in S
                        mask=sum(1<<(i-1) for i in E)-(1<<(e-1))+(1<<(s-1));col=keys[mask]
                        prob=min(1.,exp(lw[col]-lw[row]))/a^2
                        T[row,col]+=prob;T[row,row]+=1/a^2-prob
                    end
                end
                reversible=Diagonal(sqrt.(p))*T*Diagonal(1 ./sqrt.(p));@assert norm(reversible-reversible')<1e-10
                spectrum=sort(eigvals(Symmetric(reversible)));gap=1-spectrum[end-1];absgap=1-maximum(abs.(spectrum[1:end-1]))
                accept=sum(p.*(1 .- diag(T)))
                specific=[p[keys[sum(1<<(i-1) for i in initial(m.L,kind,MersenneTwister(0)))]] for kind in ("uniform_A","AB_checker","02_stripe","02_checker","mixed")]
                println(summary,join((m.L,beta,length(sets),z,normalizer(a,beta),abs(z-normalizer(a,beta)),accept,gap,absgap,1-(1-absgap)^N,classes...,specific...),','))
            end
        end
    end
end

function benchmark()
    m=model(18);rng=MersenneTwister(919);c=chain(m.G,initial(18,"random",rng),3.)
    sweep!(c,m.G,3.,rng)
    seconds=@elapsed for i in 1:5;sweep!(c,m.G,3.,rng);end
    rebuild=@elapsed rebuild!(c,m.G,3.)
    out=Dict("L"=>18,"beta"=>3.,"seconds_per_sweep"=>seconds/5,"seconds_rebuild"=>rebuild,"threads"=>BLAS.get_num_threads(),"proposals_per_sweep"=>648)
    open(io->TOML.print(io,out),joinpath(DATA,"local_benchmark.toml"),"w");println(out)
end

function run_all()
    sweeps=parse(Int,arg("sweeps","240"));burnin=parse(Int,arg("burnin","60"));refresh=parse(Int,arg("refresh","5"));seed=729031
    meta=Dict{String,Any}("status"=>"running","sweeps"=>sweeps,"burnin"=>burnin,"refresh_sweeps"=>refresh,"seed"=>seed,
        "protocol"=>"Sampling only. One sweep = 2a random E/S exchange proposals. Stable rank-two inverse kernel, exact reconstruction every refresh interval; no entropy or postmeasurement QR.",
        "ground_projector"=>"Exactly KAPDQMC HoneyComb120 nnK_Matrix; LAPACK.syevd V first L^2 columns. Degenerate zero-mode choices are retained and binary P/G are saved.",
        "threads"=>BLAS.get_num_threads(),"julia"=>string(VERSION),"code_sha256"=>sha(@__FILE__),
        "geometry_sha256"=>sha(normpath(joinpath(ROOT,"../../public/Geometry.jl"))),"models"=>Any[])
    manifest=joinpath(DATA,"local_experiment_manifest.toml");open(io->TOML.print(io,meta),manifest,"w")
    open(joinpath(DATA,"local_trajectories.csv"),"w") do traces
        open(joinpath(DATA,"local_periodic_probabilities.csv"),"w") do probability
            println(traces,"L,beta,start,chain_seed,sweep,post_burnin,acceptance,imbalance,n02_fraction,log_weight,log_probability,site_memory,inverse_drift_before_refresh,logweight_drift_before_refresh")
            println(probability,"L,beta,pattern,class_id,log_weight,log_probability,log10_probability")
            for L in (2,6,12,18)
                m=model(L);L==2 && exact_small(m)
                push!(meta["models"],Dict("L"=>L,"Ns"=>m.N,"rank"=>m.a,"zero_modes"=>count(e->abs(e)<1e-10,m.energies),
                    "G_sha256"=>sha(joinpath(DATA,"local_projector_L$(L).bin")),"P_sha256"=>sha(joinpath(DATA,"local_orbitals_L$(L).bin")),
                    "projector_error"=>norm(m.G*m.G-m.G),"orthogonality_error"=>norm(m.P'*m.P-I)))
                L==2 && continue
                for beta in (0.,.5,1.,2.,3.)
                    z=normalizer(m.a,beta)
                    for kind in ("uniform_A","AB_checker","02_stripe","02_checker","mixed")
                        E=initial(L,kind,MersenneTwister(0));lw=logweight(m.G,E,beta);_,n02,_=metrics(E,m.N,E)
                        println(probability,join((L,beta,kind,n02==0 ? 1 : n02==1 ? 2 : 3,lw,lw-z,(lw-z)/log(10)),','))
                    end
                    for (k,kind) in enumerate(("uniform_A","02_stripe","random"))
                        s=seed+L*10000+round(Int,beta*100)*10+k;rng=MersenneTwister(s)
                        E=initial(L,kind,rng);c=chain(m.G,E,beta)
                        for sweep in 0:sweeps
                            acc=sweep==0 ? NaN : sweep!(c,m.G,beta,rng)
                            inverse_drift=0.;logdrift=0.
                            if sweep>0 && sweep%refresh==0;inverse_drift,logdrift=rebuild!(c,m.G,beta);end
                            imb,n02,memory=metrics(c.E,m.N,E)
                            println(traces,join((L,beta,kind,s,sweep,sweep>burnin,acc,imb,n02,c.logw,c.logw-z,memory,inverse_drift,logdrift),','))
                        end
                        flush(traces);println("done L=$L beta=$beta start=$kind sweeps=$sweeps")
                    end
                end
            end
        end
    end
    meta["status"]="complete";meta["files"]=Dict(basename(path)=>sha(path) for path in readdir(DATA;join=true) if startswith(basename(path),"local_") && path!=manifest)
    open(io->TOML.print(io,meta),manifest,"w")
end

mode=arg("mode","benchmark")
mode=="benchmark" ? benchmark() : mode=="run" ? run_all() : error("unknown mode")
