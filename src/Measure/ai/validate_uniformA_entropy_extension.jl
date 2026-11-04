# Small-size independent validation only; never modifies existing raw datasets.
using LinearAlgebra, SHA, Dates, TOML, Printf
include(joinpath(@__DIR__,"fermion_experiment.jl"))
include(joinpath(@__DIR__,"real_entropy_extension.jl"))
include(joinpath(@__DIR__,"uniformA_entropy_extension.jl"))
import .FermionExperiment as FE
import .RealEntropyExtension as RE
import .UniformAEntropyExtension as UH

function main()
    BLAS.set_num_threads(4)
    out=joinpath(@__DIR__,"analysis_uniformA_dense")
    mkpath(out)
    fields=["L","theta","beta","full_n","half_n","S2_half","S2_FE_eigen","S2_real_cholesky",
        "half_minus_FE","half_minus_real","max_structure_error","max_imaginary","real_part_entropy_bound",
        "block_transform_error","paired_spectrum_error","half_svd_entropy_error","seconds_half","seconds_FE","seconds_real"]
    csvpath=joinpath(out,"half_entropy_validation.csv")
    maximum_error=0.0; maximum_structure=0.0; maximum_block=0.0; maximum_pair=0.0; maximum_svd=0.0
    count=0
    started=time()
    open(csvpath,"w") do io
        println(io,join(fields,','))
        for L in (18,36)
            pre=FE.prepare_pattern(FE.BlochModel(L,1),Bool[1,0])
            for theta in (0.0,4.0,Inf)
                state=FE.evolve(pre,theta)
                for beta in (0.0,0.3,1.0,2.0)
                    measured=FE.measure(state,beta)
                    t=time(); s_half,check=UH.entropy_rectangle_half(measured;with_checks=true);t_half=time()-t
                    t=time(); s_FE=FE.entropy_rectangle(measured);t_FE=time()-t
                    t=time(); s_real=RE.entropy_rectangle_real(measured);t_real=time()-t
                    indices=FE.rectangle_indices(L)
                    C=real.(FE.correlation_matrix(measured,indices))
                    p=size(C,1)÷2
                    # Independent full-size basis permutation A followed by reversed B.
                    permutation=vcat(collect(1:2:2p),reverse(collect(2:2:2p)))
                    transformed=C[permutation,permutation]-0.5I
                    M,_=UH.build_half_matrix(measured)
                    Z,Y=real.(M),imag.(M)
                    block=[Z Y; Y -Z]
                    block_error=maximum(abs,transformed-block)
                    singular=svdvals(M)
                    paired=sort(vcat(singular,-singular))
                    spectrum=eigvals(Symmetric(transformed))
                    pair_error=maximum(abs,spectrum-paired)
                    s_svd=-2sum(log.(.5 .+ 2singular.^2))
                    error=max(abs(s_half-s_FE),abs(s_half-s_real))
                    @assert error<1e-8 (L,theta,beta,error)
                    @assert block_error<1e-10 (L,theta,beta,block_error)
                    @assert pair_error<1e-10 (L,theta,beta,pair_error)
                    @assert abs(s_svd-s_half)<1e-8 (L,theta,beta,s_svd,s_half)
                    maximum_error=max(maximum_error,error);maximum_structure=max(maximum_structure,check.max_structure_error)
                    maximum_block=max(maximum_block,block_error);maximum_pair=max(maximum_pair,pair_error)
                    maximum_svd=max(maximum_svd,abs(s_svd-s_half))
                    row=(L,theta,beta,2p,p,s_half,s_FE,s_real,s_half-s_FE,s_half-s_real,
                        check.max_structure_error,check.max_imaginary,check.real_part_entropy_bound,
                        block_error,pair_error,s_svd-s_half,t_half,t_FE,t_real)
                    println(io,join(row,','));flush(io)
                    count+=1
                    @printf("%d/24 L=%d theta=%g beta=%g error=%.3g symmetry=%.3g\n",count,L,theta,beta,error,check.max_structure_error)
                end
            end
        end
    end
    files=["uniformA_entropy_extension.jl","validate_uniformA_entropy_extension.jl","fermion_experiment.jl","real_entropy_extension.jl"]
    receipt=Dict("status"=>"PASS","cases"=>count,"L"=>[18,36],"theta"=>["0","4","Inf"],"beta"=>[0.0,.3,1.0,2.0],
        "generated_utc"=>string(now(Dates.UTC)),"elapsed_seconds"=>time()-started,"julia"=>string(VERSION),
        "blas_threads"=>BLAS.get_num_threads(),"max_entropy_error"=>maximum_error,
        "max_structure_error"=>maximum_structure,"max_block_transform_error"=>maximum_block,
        "max_paired_spectrum_error"=>maximum_pair,"max_svd_entropy_error"=>maximum_svd,
        "source_sha256"=>Dict(name=>bytes2hex(sha256(read(joinpath(@__DIR__,name)))) for name in files),
        "data_sha256"=>bytes2hex(sha256(read(csvpath))),
        "scope"=>"Small-size identity and numerical validation; recorded timings include compilation and are not a large-size benchmark. Existing kernels/data unchanged.")
    open(io->TOML.print(io,receipt),joinpath(out,"half_entropy_validation.toml"),"w")
    println("PASS max entropy error=$maximum_error; paired spectrum error=$maximum_pair")
end
abspath(PROGRAM_FILE)==abspath(@__FILE__) && main()
