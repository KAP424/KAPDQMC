# Exact stationary finite-N variance for the L=2 fixed-half swap chain.
# No new Monte Carlo samples. One recorded step is Nsites elementary proposals.
using LinearAlgebra, SHA, TOML, Dates
BLAS.set_num_threads(1)
const ROOT=@__DIR__
const DATA=joinpath(ROOT,"data","exact_born")
const OBS=("S_A","S_B","S_C","A_minus_B","D")
sha(path)=bytes2hex(sha256(read(path)))
function rows(path)
    lines=readlines(path); header=split(first(lines),',';keepempty=true)
    [Dict(zip(header,split(line,',';keepempty=true))) for line in lines[2:end]]
end
csv(io,x)=println(io,join(string.(x),','))

function main()
    started=time()
    core=TOML.parsefile(joinpath(DATA,"exact_born_manifest.toml"))
    n=core["local_samples_per_chain"]; chains=core["local_chains"]
    summary=rows(joinpath(DATA,"exact_born_summary.csv"))
    outfile=joinpath(DATA,"exact_born_stationary_local_variance.csv")
    files=Dict{String,String}(); checks=Dict{String,Any}[]
    open(outfile,"w") do out
        csv(out,("L","beta","observable","n_per_chain","independent_chains","exact_variance",
                 "stationary_variance_single_chain_mean","stationary_SE_pooled_mean","iid_SE_same_total_n",
                 "local_mean","exact_mean","local_difference","local_empirical_SE","local_stationary_z",
                 "spectral_gap_per_sweep","asymptotic_g"))
        for beta in (0.,.5,1.,2.,3.)
            filename="exact_born_L2_beta$(replace(string(beta),'.'=>'p'))_trajectories.csv"
            path=joinpath(DATA,filename); rel=replace(relpath(path,ROOT),'\\'=>'/')
            @assert sha(path)==core["files"][rel]
            files[rel]=sha(path)
            data=rows(path); count=length(data); @assert count==70
            keys=[row["E_indices"] for row in data]
            lookup=Dict(key=>i for (i,key) in enumerate(keys))
            configs=[parse.(Int,split(key,';')) for key in keys]
            probability=[parse(Float64,row["probability"]) for row in data]
            logweight=[parse(Float64,row["log_weight"]) for row in data]
            transition=zeros(count,count)
            for (i,E) in enumerate(configs)
                S=setdiff(1:8,E)
                for e in E,s in S
                    F=sort(vcat(filter(!=(e),E),s)); j=lookup[join(F,';')]
                    transition[i,j]+=min(1.,exp(logweight[j]-logweight[i]))/16
                end
                transition[i,i]=1-sum(view(transition,i,:))
            end
            @assert minimum(transition)>-1e-14
            stationary_error=norm(transition'*probability-probability,Inf)
            sweep=transition^8
            sqrtp=sqrt.(probability)
            reversible=(sqrtp.*sweep)./sqrtp'
            symmetry_error=norm(reversible-reversible',Inf)
            @assert stationary_error<1e-12 && symmetry_error<1e-10
            eigen=eigen!(Symmetric((reversible+reversible')/2))
            lambda=clamp.(eigen.values,-1.,1.)
            gap=1-lambda[end-1]
            # Direct finite geometric sum avoids cancellation for lambda near one.
            factors=fill(Float64(n),count); powers=copy(lambda)
            for lag in 1:n-1
                factors .+= (2(n-lag)).*powers
                powers .*=lambda
            end
            factors./=n^2
            for obs in OBS
                values=[parse(Float64,row[obs]) for row in data]
                mu=dot(probability,values)
                centered=values.-mu
                variance=dot(probability,centered.^2)
                amplitudes=eigen.vectors'*(sqrtp.*centered)
                single_var=dot(amplitudes.^2,factors)
                g=variance>0 ? sum(amplitudes[k]^2*(1+lambda[k])/(1-lambda[k]) for k in 1:count-1)/variance : 1.
                if maximum(values)-minimum(values)<1e-12
                    variance=0.; single_var=0.; g=1.
                end
                localrow=only(filter(row->row["L"]=="2" && parse(Float64,row["beta"])==beta && row["method"]=="local" && row["observable"]==obs,summary))
                localmean=parse(Float64,localrow["mean"]); localse=parse(Float64,localrow["SE"])
                se=sqrt(max(0.,single_var)/chains); iidse=sqrt(variance/(n*chains))
                z=se>0 ? (localmean-mu)/se : ""
                csv(out,(2,beta,obs,n,chains,variance,single_var,se,iidse,localmean,mu,localmean-mu,localse,z,gap,g))
            end
            push!(checks,Dict("beta"=>beta,"states"=>count,"stationary_error_inf"=>stationary_error,
                 "detailed_balance_symmetry_error_inf"=>symmetry_error,"spectral_gap_per_sweep"=>gap))
        end
    end
    for path in [@__FILE__,outfile,joinpath(DATA,"exact_born_manifest.toml"),joinpath(DATA,"exact_born_summary.csv")]
        files[replace(relpath(path,ROOT),'\\'=>'/')]=sha(path)
    end
    receipt=Dict("status"=>"complete","audit_status"=>"PASS","created_at"=>string(now()),"elapsed_seconds"=>time()-started,
        "formula"=>"Var(mean_n)=n^-2 [n gamma0+2 sum_{k=1}^{n-1}(n-k)gamma_k]; reversible 70-state exact transition matrix, one sweep=8 proposals",
        "limitation"=>"Stationary reference only; does not include finite-burnin bias. Independent chains are pooled with equal length.",
        "files"=>files,"checks"=>checks)
    open(joinpath(DATA,"exact_born_stationary_local_variance.toml"),"w") do io; TOML.print(io,receipt); end
    println("PASS stationary variance elapsed=$(time()-started)s")
end
if abspath(PROGRAM_FILE)==@__FILE__; main(); end
