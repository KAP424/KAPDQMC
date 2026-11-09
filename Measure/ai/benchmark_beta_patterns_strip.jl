using LinearAlgebra, Dates, TOML, SHA
include(joinpath(@__DIR__,"fermion_experiment.jl"))
import .FermionExperiment as FE
include(joinpath(@__DIR__,"experiment_patterns.jl"))
include(joinpath(@__DIR__,"beta_patterns_strip_extension.jl"))
function main()
    L=180
    p=only(filter(p->p.name=="random_half_3",experiment_patterns()))
    BLAS.set_num_threads(1)
    pre=FE.prepare_pattern(FE.BlochModel(L,p.m),p.occupied)
    state=FE.measure(FE.evolve(pre,4.),1.)
    small=FE.measure(FE.evolve(FE.prepare_pattern(FE.BlochModel(18,p.m),p.occupied),4.),1.)
    rows=Dict{String,Any}[]
    reference=nothing
    for (method,threads) in (("original",8),("original",1),("folded",1),("folded",8))
        BLAS.set_num_threads(threads)
        f=method=="original" ? FE.entropy_strip : BetaPatternsStripExtension.entropy_strip_folded
        f(small);f(small;orientation=:y)
        for repeat in 1:2
            GC.gc(true)
            start=time();sa=f(state);sc=f(state;orientation=:y);seconds=time()-start
            isnothing(reference) && (reference=(sa,sc))
            error=max(abs(sa-reference[1]),abs(sc-reference[2]))
            error<1e-9 || Base.error("strip method/thread discrepancy")
            push!(rows,Dict("pattern"=>p.name,"L"=>L,"theta"=>4.,"beta"=>1.,"method"=>method,"threads"=>threads,
                "repeat"=>repeat,"S_A"=>sa,"S_C"=>sc,"seconds"=>seconds,"absolute_error"=>error))
            println(method," threads=",threads," repeat=",repeat," seconds=",seconds," error=",error);flush(stdout)
        end
    end
    receipt=Dict("status"=>"complete","finished_utc"=>string(now(Dates.UTC)),"observations"=>rows,
        "blas_config"=>sprint(show,BLAS.get_config()),"julia"=>string(VERSION),
        "script_sha256"=>bytes2hex(sha256(read(@__FILE__))),
        "strip_extension_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"beta_patterns_strip_extension.jl")))))
    open(io->TOML.print(io,receipt),joinpath(@__DIR__,"data","beta_patterns_strip_benchmark.toml"),"w")
end
main()
