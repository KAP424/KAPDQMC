# Independent targeted validation of exact channel/variance identities.
# No Markov updates, scaling fits, outcome-dependent stopping, or pilot reuse.
using LinearAlgebra, Random, SHA, TOML, Dates
BLAS.set_num_threads(1)
include("StratifiedBorn.jl")
using .StratifiedBorn
include("../original_swap_scaling.jl")
using .OriginalSwapScaling
include("../ExactBorn.jl")
using .ExactBorn
const HERE=@__DIR__
const OUT=joinpath(HERE,"iid_targeted")
const PLAN=joinpath(OUT,"plan.toml")
const HEADER="L,beta,batch,sample,seed,j,F,E,record_sha256,SA,SB,SC,D,VA,VB,VC,DV,VA_full,VB_full,VC_full,DV_full,DV_subgroup,NA,NB,NC,channel_probe,channel_residual,Vlaw_A_residual,Vlaw_B_residual,Vlaw_C_residual,sampling_seconds,entropy_seconds,diagnostic_seconds"
sha(p)=bytes2hex(sha256(read(p)))
rel(p)=replace(relpath(p,HERE),'\\'=>'/')
save_toml(p,d)=open(io->TOML.print(io,d),p,"w")
arg(name,default)=let a=filter(x->startswith(x,"--$name="),ARGS);isempty(a) ? default : split(last(a),'=';limit=2)[2];end

function sources()
    Dict(rel(p)=>sha(p) for p in [@__FILE__,joinpath(HERE,"StratifiedBorn.jl"),joinpath(HERE,"../BornSampler.jl"),
        joinpath(HERE,"../original_swap_scaling.jl"),joinpath(HERE,"../ExactBorn.jl"),
        normpath(joinpath(HERE,"../../../public/Geometry.jl"))])
end

function prepare()
    ispath(OUT) && error("targeted directory exists; preserve existing evidence")
    mkpath(OUT);models=Dict{String,Any}[];jobs=Dict{String,Any}[]
    for L in (18,12,6)
        m=ground_model(L;state=:staggered,mu=1e-8)
        paths=save_model(m,joinpath(OUT,"models");prefix="iid_targeted")
        metadata=only(filter(p->endswith(p,".toml"),paths))
        push!(models,Dict("L"=>L,"path"=>rel(metadata),"files"=>Dict(rel(p)=>sha(p) for p in paths)))
        for (bi,beta) in enumerate((1.,3.)),batch in 1:4
            seed=729_102_900+10000L+100bi+batch
            id="L$(L)_beta$(Int(beta))_batch$(batch)"
            push!(jobs,Dict("id"=>id,"L"=>L,"beta"=>beta,"batch"=>batch,"seed"=>seed,"records"=>512,"model"=>rel(metadata)))
        end
    end
    evidence=[joinpath(HERE,"validation/receipt.toml"),joinpath(HERE,"validation/fluctuation_kernel.json"),
        joinpath(HERE,"pilot_analysis/summary.json"),joinpath(HERE,"pilot_analysis/allocation_comparison.json"),
        joinpath(HERE,"pilot_analysis/allocation_comparison.csv")]
    plan=Dict("stage"=>"independent_targeted_production","created_at"=>string(now()),"models"=>models,"jobs"=>jobs,
        "source_sha256"=>sources(),"evidence_sha256"=>Dict(rel(p)=>sha(p) for p in evidence),
        "records_per_parameter"=>2048,"independent_batches"=>4,"mu"=>1e-8,"blas_threads"=>1,
        "estimator"=>"Ordinary iid mean; fresh independent auxiliary F and j for every sample, j has full analytic distribution",
        "geometry"=>"Origin p=1/3 A=w x L,B=w x 2w,C=L x w. Full/subgroup translated number-variance D measured separately.",
        "core_checks"=>"channel_probe is projection of offdiagonal G onto G0, expectation kappa. Vlaw_R=V_R+(N_R-N_R0)^2-V_R0 has expectation zero.",
        "kappa"=>"1-(1-exp(-2beta))^2 Z_(a-1)/Z_a; exact on the fixed-a particle sector",
        "sampling_uncertainty"=>"Empirical iid SE and four independent batches; rare tails may defeat finite-sample uncertainty estimates",
        "fixed_budget"=>"L6,12,18; beta1,3;4 batches each512. No data-dependent stopping, no pilot samples in production, no seed replacement.",
        "purpose"=>"Targeted numerical verification of exact channel, variance decomposition and bounds; not an entropy corner-coefficient scaling fit")
    save_toml(PLAN,plan)
    println("PREPARED jobs=$(length(jobs)) records=$(sum(j["records"] for j in jobs)) plan_sha256=$(sha(PLAN))")
end

function validate_plan()
    plan=TOML.parsefile(PLAN)
    for mapping in (plan["source_sha256"],plan["evidence_sha256"]), (p,h) in mapping
        sha(joinpath(HERE,p))==h || error("frozen dependency changed: $p")
    end
    for m in plan["models"],(p,h) in m["files"];sha(joinpath(HERE,p))==h || error("model changed");end
    plan
end

function load_model(meta)
    d=TOML.parsefile(meta);N=d["Nsites"];n=d["Nparticles"]
    arrays=Dict{String,Any}()
    for (name,shape) in (("Q",(N,n)),("G",(N,N)),("energies",(N,)))
        file=only(filter(f->endswith(f,"_$(name).bin"),collect(keys(d["files"]))));path=joinpath(dirname(meta),file)
        sha(path)==d["files"][file] || error("model hash changed")
        x=Array{Float64}(undef,shape...);open(io->read!(io,x),path,"r");arrays[name]=x
    end
    GroundModel(d["L"],N,n,arrays["Q"],arrays["G"],arrays["energies"],:staggered,d["mu"],d["null_singular_count"],d["zero_singular_tolerance"],d["orthogonality_error"],d["ground_energy_error"],d["eigenvector_residual"])
end

function geometry(m)
    L=m.L;N=m.N;w=L÷3
    x=[((i-1)÷2)%L for i in 1:N];y=[((i-1)÷2)÷L for i in 1:N]
    Kx=zeros(N,N);Ky=zeros(N,N);Ksub=zeros(N,N)
    for j in 1:N,i in 1:N
        dx=mod(x[i]-x[j],L);dy=mod(y[i]-y[j],L)
        Kx[i,j]=min(dx,L-dx,w)/L;Ky[i,j]=min(dy,L-dy,w)/L
        Ksub[i,j]=((x[i]÷w!=x[j]÷w)&&(y[i]÷w!=y[j]÷w))/9
    end
    regions=Tuple(values(region_indices(L)))
    N0=[sum(diag(m.G)[r]) for r in regions]
    V0=[N0[k]-sum(abs2,m.G[r,r]) for (k,r) in enumerate(regions)]
    off=copy(m.G);off[diagind(off)].=0.
    (;Kx,Ky,Kfull=Kx.*Ky,Ksub,N0,V0,off,offnorm=sum(abs2,off))
end

function diagnostics!(G,Gsquare,work,kernel,kappa)
    mul!(G,work.Q,work.Q');Gsquare.=abs2.(G)
    V=[length(work.regions[k])/4-sum(abs2,work.correlation[k]) for k in 1:3]
    minimum(V)>=-1e-9 || error("negative number variance")
    DV=(2V[1]+V[3])/3-V[2]
    VAfull=dot(Gsquare,kernel.Kx);VCfull=dot(Gsquare,kernel.Ky)
    DVfull=dot(Gsquare,kernel.Kfull);DVsub=dot(Gsquare,kernel.Ksub)
    VBfull=(2VAfull+VCfull)/3-DVfull
    nums=[sum(G[i,i] for i in r) for r in work.regions]
    probe=dot(G,kernel.off)/kernel.offnorm
    laws=V.+(nums.-kernel.N0).^2 .-kernel.V0
    [V...,DV,VAfull,VBfull,VCfull,DVfull,DVsub,nums...,probe,probe-kappa,laws...]
end

function reference_entropy(m,E,beta,regions)
    P=copy(m.Q);P[E,:].*=exp(-2beta);Q=Matrix(qr(P).Q)[:,1:m.n]
    v=[ExactBorn.regional_entropy(Q,r,m.N) for r in regions]
    [v...,(2v[1]+v[3])/3-v[2]]
end

function valid_done(job,planhash)
    receipt=joinpath(OUT,job["id"]*"_receipt.toml")
    isfile(receipt) || return false
    r=TOML.parsefile(receipt)
    r["status"]=="complete" && r["plan_sha256"]==planhash && r["job"]==job || error("invalid completed receipt")
    for (p,h) in r["files"];sha(joinpath(HERE,p))==h || error("completed file changed");end
    true
end

function run(worker)
    worker in 1:4 || error("worker must be1..4")
    plan=validate_plan();planhash=sha(PLAN)
    for job in plan["jobs"]
        job["batch"]==worker || continue
        if valid_done(job,planhash);println("SKIP $(job["id"])");continue;end
        stem=joinpath(OUT,job["id"]);data=stem*".csv";receipt=stem*"_receipt.toml"
        ispath(data) || ispath(data*".partial") ? error("refuse overwrite existing job output") : nothing
        m=load_model(joinpath(HERE,job["model"]));work=EntropyWorkspace(m);kernel=geometry(m)
        s=StratifiedSampler(m.Q,job["beta"]);rng=MersenneTwister(job["seed"])
        t=exp(-2job["beta"])
        kappa=1-(1-t)^2*exp(BornSampler.log_normalizer(m.n-1,job["beta"])-s.log_Z)
        G=zeros(m.N,m.N);Gsquare=similar(G);sumG=zeros(m.N,m.N);sumGsquare=zeros(m.N,m.N)
        # Fixed discarded calls separate compilation from per-record timings.
        warm=sample_stratum(s,0,m.n;rng=MersenneTwister(100+worker))
        entropy!(work,m,warm.E,job["beta"]);diagnostics!(G,Gsquare,work,kernel,kappa)
        reference_entropy(m,warm.E,job["beta"],work.regions)
        maxerror=0.;maxentropybound=0.;sampling_time=0.;entropy_time=0.;diag_time=0.;wall=time();jcounts=zeros(Int,m.n+1)
        open(data*".partial","w") do io
            println(io,HEADER)
            for k in 1:job["records"]
                start=time_ns();draw=sample_stratum(s,0,m.n;rng);ts=(time_ns()-start)/1e9
                start=time_ns();entropy=entropy!(work,m,draw.E,job["beta"]);te=(time_ns()-start)/1e9
                y=[entropy.SA,entropy.SB,entropy.SC,entropy.D]
                start=time_ns();v=diagnostics!(G,Gsquare,work,kernel,kappa);td=(time_ns()-start)/1e9
                all(isfinite,y) && all(isfinite,v) || error("nonfinite observable")
                violation=max(maximum(2v[1:3]-y[1:3]),maximum(y[1:3]-4log(2)*v[1:3]),0.)
                maxentropybound=max(maxentropybound,violation);maxentropybound<1e-9 || error("single-state entropy/variance bound failed")
                if k<=2
                    maxerror=max(maxerror,maximum(abs.(reference_entropy(m,draw.E,job["beta"],work.regions)-y)))
                    maxerror<1e-9 || error("entropy reference disagreement")
                end
                sumG.+=G;sumGsquare.+=Gsquare;jcounts[draw.j+1]+=1
                sampling_time+=ts;entropy_time+=te;diag_time+=td
                row=(job["L"],job["beta"],job["batch"],k,job["seed"],draw.j,join(draw.F,';'),join(draw.E,';'),record_digest(draw.F,draw.E,draw.j),y...,v...,ts,te,td)
                println(io,join(row,','))
                if k%128==0;flush(io);println("PROGRESS $(job["id"]) $k/$(job["records"])");flush(stdout);end
            end
        end
        mv(data*".partial",data)
        meanGpath=stem*"_meanG.bin";meanG2path=stem*"_meanGsquare.bin"
        open(io->write(io,sumG/job["records"]),meanGpath,"w")
        open(io->write(io,sumGsquare/job["records"]),meanG2path,"w")
        record=Dict("status"=>"complete","job"=>job,"plan_sha256"=>planhash,"created_at"=>string(now()),
            "elapsed_seconds"=>time()-wall,"sampling_seconds"=>sampling_time,"entropy_seconds"=>entropy_time,"diagnostic_seconds"=>diag_time,
            "maximum_reference_entropy_error"=>maxerror,"maximum_single_state_bound_violation"=>maxentropybound,
            "kappa"=>kappa,"N0"=>kernel.N0,"V0"=>kernel.V0,"DV0_full"=>dot(abs2.(m.G),kernel.Kfull),
            "j_counts"=>jcounts,"records"=>job["records"],"files"=>Dict(rel(p)=>sha(p) for p in (data,meanGpath,meanG2path)),
            "matrix_format"=>"Float64 native-endian column-major Nsites x Nsites","julia_version"=>string(VERSION),"blas_threads"=>BLAS.get_num_threads())
        save_toml(receipt,record);valid_done(job,planhash) || error("receipt validation failed")
        println("DONE $(job["id"]) seconds=$(round(time()-wall;digits=2))");flush(stdout)
    end
end

function finalize()
    plan=validate_plan();planhash=sha(PLAN);files=Dict{String,String}();receipts=String[]
    for job in plan["jobs"]
        valid_done(job,planhash) || error("incomplete job $(job["id"])")
        p=joinpath(OUT,job["id"]*"_receipt.toml");r=TOML.parsefile(p)
        merge!(files,r["files"]);files[rel(p)]=sha(p);push!(receipts,rel(p))
    end
    for model in plan["models"];merge!(files,model["files"]);end
    files[rel(PLAN)]=planhash
    path=joinpath(OUT,"manifest.toml");isfile(path) && error("manifest already exists")
    save_toml(path,Dict("status"=>"complete","created_at"=>string(now()),"files"=>files,"receipts"=>receipts,
        "records"=>sum(j["records"] for j in plan["jobs"]),"plan_sha256"=>planhash,
        "source_sha256"=>plan["source_sha256"],"evidence_sha256"=>plan["evidence_sha256"]))
    println("COMPLETE $(sha(path))")
end

if abspath(PROGRAM_FILE)==@__FILE__
    if "--prepare" in ARGS;prepare()
    elseif "--finalize" in ARGS;finalize()
    elseif startswith(arg("worker","none"),"none");error("Use --prepare, --worker=N, or --finalize")
    else;run(parse(Int,arg("worker","0")));end
end
