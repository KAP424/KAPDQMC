# Exact fixed-half Born reference for the new staggered-mass production state.
using LinearAlgebra, SHA, TOML
BLAS.set_num_threads(1)
include(joinpath(@__DIR__,"..","original_swap_scaling.jl"))
include(joinpath(@__DIR__,"..","ExactBorn.jl"))
using .OriginalSwapScaling

const EXACT_MASS_DIR=normpath(joinpath(@__DIR__,"..","data","born_scaling"))
const EXACT_MASS_BETAS=(0.5,1.5,3.0)
const EXACT_MASS_OBS=("S_A","S_B","S_C","D")
file_sha(p)=bytes2hex(sha256(read(p)))
write_csv_row(io,row)=println(io,join(string.(row),','))

function tail_functional(p,h,alpha;absolute_order=false)
    key=absolute_order ? abs.(h) : h
    order=sortperm(key;rev=true)
    left=alpha; contribution=0.; absolute_contribution=0.; second_moment=0.
    boundary=0; boundary_used=0.; visited=0
    for index in order
        used=min(left,p[index])
        contribution+=used*h[index]
        absolute_contribution+=used*abs(h[index])
        second_moment+=used*h[index]^2
        left-=used; visited+=1
        if left<=1e-16
            boundary=index; boundary_used=used; break
        end
    end
    @assert boundary>0 && abs(left)<1e-14
    threshold=key[boundary]
    # Also supply a realizable closed event. Include numerical symmetry ties.
    event=key.>=threshold-1e-12
    (;alpha,threshold,contribution,absolute_contribution,second_moment,
      fractional_configurations_visited=visited,boundary_configuration=boundary,
      boundary_probability=p[boundary],boundary_used_probability=boundary_used,
      closed_event_probability=sum(p[event]),closed_event_contribution=sum(p[event].*h[event]),
      closed_event_absolute_contribution=sum(p[event].*abs.(h[event])),
      closed_event_configurations=count(event))
end

function main_exact_mass()
    mkpath(EXACT_MASS_DIR)
    sources=[@__FILE__,joinpath(@__DIR__,"..","original_swap_scaling.jl"),joinpath(@__DIR__,"..","ExactBorn.jl")]
    source_hashes=Dict(abspath(p)=>file_sha(p) for p in sources)
    model=ground_model(3;state=:staggered,mu=1e-8)
    regions=region_indices(3)
    @assert Tuple(length.(values(regions)))==(6,4,6)
    transform=[1. 0 0;0 1 0;0 0 1;2/3 -1 1/3]
    summarypath=joinpath(EXACT_MASS_DIR,"exact_L3_mu1e-8.csv")
    covariancepath=joinpath(EXACT_MASS_DIR,"exact_L3_mu1e-8_covariance.csv")
    tailpath=joinpath(EXACT_MASS_DIR,"exact_L3_mu1e-8_tails.csv")
    outputs=[summarypath,covariancepath,tailpath]
    max_normalization_error=0.; max_covariance_error=0.; max_mean_error=0.
    open(summarypath,"w") do summary
        write_csv_row(summary,("L","mu","beta","ensemble","observable","mean","variance","stddev","configurations",
            "log_normalizer","normalizer","probability_sum","probability_mass_before_normalization","normalization_error","weight_sum_error"))
        open(covariancepath,"w") do covout
            write_csv_row(covout,("L","mu","beta","observable_i","observable_j","covariance"))
            open(tailpath,"w") do tailout
                write_csv_row(tailout,("L","mu","beta","observable","ranking","tail_probability_requested","threshold",
                    "fractional_tail_mean_contribution","fractional_tail_absolute_contribution","fractional_tail_second_moment",
                    "mean_fraction","absolute_mean_fraction","configurations_visited","boundary_configuration","boundary_probability",
                    "boundary_used_probability","closed_event_probability","closed_event_mean_contribution",
                    "closed_event_absolute_contribution","closed_event_configurations"))
                for beta in EXACT_MASS_BETAS
                    exact=ExactBorn.born_trajectory_entropy(model.Q,beta,regions;
                        ensemble=:fixed_half,keep_trajectories=true,max_configs=50_000)
                    @assert exact.nconfigurations==48_620
                    p=exact.probabilities
                    h=exact.entropies*transform'
                    mu=vec(p'*h)
                    centered=h.-mu'
                    covariance=centered'*(p.*centered)
                    meanerr=maximum(abs.(mu-transform*exact.mean_entropy))
                    coverr=maximum(abs.(covariance-transform*exact.entropy_covariance*transform'))
                    @assert max(meanerr,coverr)<1e-11
                    max_mean_error=max(max_mean_error,meanerr)
                    max_covariance_error=max(max_covariance_error,coverr)
                    max_normalization_error=max(max_normalization_error,exact.normalization_error,exact.weight_sum_error)
                    for j in 1:4
                        write_csv_row(summary,(3,1e-8,beta,"fixed_half",EXACT_MASS_OBS[j],mu[j],covariance[j,j],sqrt(covariance[j,j]),
                            exact.nconfigurations,exact.expected_log_weight_sum,exp(exact.expected_log_weight_sum),exact.probability_sum,
                            exact.probability_mass_before_normalization,exact.normalization_error,exact.weight_sum_error))
                        for k in 1:4
                            write_csv_row(covout,(3,1e-8,beta,EXACT_MASS_OBS[j],EXACT_MASS_OBS[k],covariance[j,k]))
                        end
                        for absolute_order in (j==4 ? (false,true) : (false,)),alpha in (0.01,0.001)
                            tail=tail_functional(p,h[:,j],alpha;absolute_order)
                            absmu=sum(p.*abs.(h[:,j]))
                            write_csv_row(tailout,(3,1e-8,beta,EXACT_MASS_OBS[j],absolute_order ? "absolute_observable" : "observable",
                                alpha,tail.threshold,tail.contribution,tail.absolute_contribution,tail.second_moment,
                                tail.contribution/mu[j],tail.absolute_contribution/absmu,tail.fractional_configurations_visited,
                                tail.boundary_configuration,tail.boundary_probability,tail.boundary_used_probability,
                                tail.closed_event_probability,tail.closed_event_contribution,tail.closed_event_absolute_contribution,
                                tail.closed_event_configurations))
                        end
                    end
                    tag=replace(string(beta),"."=>"p")
                    path=joinpath(EXACT_MASS_DIR,"exact_L3_mu1e-8_beta$(tag)_trajectories.csv")
                    push!(outputs,path)
                    open(path,"w") do io
                        write_csv_row(io,("L","mu","beta","configuration","E_indices","probability","log_weight",EXACT_MASS_OBS...))
                        for i in eachindex(p)
                            write_csv_row(io,(3,1e-8,beta,i,join(exact.configurations[i],';'),p[i],exact.log_weights[i],h[i,:]...))
                        end
                    end
                    println("Exact L3 mu1e-8 beta=$beta SA=$(mu[1]) D=$(mu[4]) varD=$(covariance[4,4])")
                    flush(stdout)
                end
            end
        end
    end
    @assert all(file_sha(p)==v for (p,v) in source_hashes)
    root=normpath(joinpath(@__DIR__,".."))
    files=Dict(replace(relpath(p,root),'\\'=>'/')=>file_sha(p) for p in outputs)
    receipt=Dict("status"=>"PASS","L"=>3,"mu"=>1e-8,"betas"=>collect(EXACT_MASS_BETAS),
        "ensemble"=>"fixed_half","configurations_per_beta"=>48_620,"sampling"=>"None: exhaustive enumeration",
        "region_sizes"=>[6,4,6],"D_definition"=>"(2*S_A+S_C)/3-S_B, formed per trajectory before averaging",
        "ground_orbital_sha256"=>bytes2hex(sha256(reinterpret(UInt8,vec(model.Q)))),
        "max_normalization_error"=>max_normalization_error,"max_covariance_transform_error"=>max_covariance_error,
        "max_mean_transform_error"=>max_mean_error,
        "tail_definition"=>"Sort outcomes by the indicated observable (also abs(D) for D), then accumulate BORN probability from the largest value down to exactly alpha=0.01 or 0.001. A fractional final atom fills exactly alpha. This is a weighted upper-tail functional, not the top alpha fraction of configuration counts. Mean contribution retains the sign of D; absolute contribution is separately reported. Closed-event columns include ALL outcomes above threshold-1e-12, including ties, and report their actual probability (which can exceed alpha).",
        "scope"=>"Exact small-system reference for the chosen staggered-mass Slater. No convergence claim for larger L and no replacement of the original swap production method.",
        "source_hashes"=>source_hashes,"files"=>files)
    open(joinpath(EXACT_MASS_DIR,"exact_L3_mu1e-8.toml"),"w") do io
        TOML.print(io,receipt;sorted=true)
    end
    println("PASS: L3 exact mass references and weighted tail functionals")
end

main_exact_mass()
