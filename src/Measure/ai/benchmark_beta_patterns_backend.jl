using LinearAlgebra, Dates, TOML, SHA
function bb_arg(name,default)
    prefix="--$name=";i=findfirst(a->startswith(a,prefix),ARGS)
    isnothing(i) ? default : ARGS[i][length(prefix)+1:end]
end
backend=bb_arg("backend","openblas")
backend in ("openblas","mkl") || error("unknown backend")
if backend=="mkl"
    @eval using MKL
end
include(joinpath(@__DIR__,"fermion_experiment.jl"))
import .FermionExperiment as FE
include(joinpath(@__DIR__,"real_entropy_extension.jl"))
include(joinpath(@__DIR__,"experiment_patterns.jl"))
BLAS.set_num_threads(parse(Int,bb_arg("threads","8")))
mode=bb_arg("mode","validate")
config=sprint(show,BLAS.get_config())
receipt=Dict{String,Any}("backend"=>backend,"blas_config"=>config,"blas_threads"=>BLAS.get_num_threads(),
    "julia"=>string(VERSION),"started_utc"=>string(now(Dates.UTC)),
    "script_sha256"=>bytes2hex(sha256(read(@__FILE__))),"observations"=>Dict{String,Any}[])
if mode=="validate"
    for name in ("one_B_2","diagonal_B_3","random_half_3")
        p=only(filter(p->p.name==name,experiment_patterns()))
        pre=FE.prepare_pattern(FE.BlochModel(18,p.m),p.occupied)
        for theta in (4.,Inf), beta in (0.,1.,2.)
            state=FE.measure(FE.evolve(pre,theta),beta)
            sa=FE.entropy_strip(state);sc=FE.entropy_strip(state;orientation=:y)
            sb,checks=RealEntropyExtension.entropy_rectangle_real(state;with_checks=true)
            reference=FE.entropy_rectangle(state)
            abs(sb-reference)<1e-9 || error("entropy method discrepancy")
            push!(receipt["observations"],Dict("pattern"=>name,"L"=>18,"theta"=>string(theta),"beta"=>beta,
                "S_A"=>sa,"S_C"=>sc,"S_B"=>sb,"D"=>(2sa+sc)/3-sb,"eigen_reference"=>reference,
                "entropy_error"=>abs(sb-reference),"projector_error"=>checks.orbital_projector_error))
        end
    end
elseif mode=="benchmark"
    L=parse(Int,bb_arg("L","108"));p=only(filter(p->p.name=="random_half_3",experiment_patterns()))
    pre=FE.prepare_pattern(FE.BlochModel(L,p.m),p.occupied)
    state=FE.measure(FE.evolve(pre,4.),1.)
    # Compile on a small physical covariance before recording representative timings.
    small=FE.measure(FE.evolve(FE.prepare_pattern(FE.BlochModel(18,p.m),p.occupied),4.),1.)
    RealEntropyExtension.entropy_rectangle_real(small)
    FE.entropy_strip(small)
    for repeat in 1:2
        GC.gc(true)
        start=time();sa=FE.entropy_strip(state);sc=FE.entropy_strip(state;orientation=:y);strip=time()-start
        start=time();sb,checks=RealEntropyExtension.entropy_rectangle_real(state;with_checks=true);rectangle=time()-start
        row=Dict("pattern"=>p.name,"L"=>L,"theta"=>"4.0","beta"=>1.,"repeat"=>repeat,
            "S_A"=>sa,"S_C"=>sc,"S_B"=>sb,"D"=>(2sa+sc)/3-sb,"strip_seconds"=>strip,
            "rectangle_seconds"=>rectangle,"total_seconds"=>strip+rectangle,
            "max_imaginary"=>checks.max_imaginary,"projector_error"=>checks.orbital_projector_error)
        push!(receipt["observations"],row)
        println(backend," L=",L," repeat=",repeat," seconds=",strip+rectangle," D=",row["D"]);flush(stdout)
    end
else
    error("unknown mode")
end
receipt["finished_utc"]=string(now(Dates.UTC));receipt["status"]="complete"
output=bb_arg("output",joinpath(@__DIR__,"data","beta_patterns_backend_$(backend)_$(mode).toml"))
open(io->TOML.print(io,receipt),output,"w")
println("backend ",backend," ",mode," complete: ",length(receipt["observations"])," observations")
