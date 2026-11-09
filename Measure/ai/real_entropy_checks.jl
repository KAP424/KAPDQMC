# Independent validation and a single optional large-row benchmark.
using LinearAlgebra, SHA, Dates
BLAS.set_num_threads(4)
include(joinpath(@__DIR__,"fermion_experiment.jl"))
import .FermionExperiment as FE
include(joinpath(@__DIR__,"experiment_patterns.jl"))
include(joinpath(@__DIR__,"real_entropy_extension.jl"))
import .RealEntropyExtension as RE
const REAL_OUT=joinpath(@__DIR__,"data")

function complex_entropy(C)
    v=clamp.(eigvals(Hermitian(C)),0.0,1.0)
    return -sum(log1p.(-2 .* v .* (1 .- v)))
end

function csvwrite(path,rows)
    open(path,"w") do io
        println(io,join(keys(first(rows)),','))
        for row in rows
            println(io,join(values(row),','))
        end
    end
end

function validate()
    p=only(filter(p->p.name=="random_half_3",experiment_patterns()))
    rows=NamedTuple[]
    for L in (18,36)
        pre=FE.prepare_pattern(FE.BlochModel(L,3),p.occupied)
        for theta in (4.0,8.0,16.0,Inf)
            state=FE.evolve(pre,theta;beta=1)
            indices=FE.rectangle_indices(L)
            cold=FE.correlation_matrix(state,indices)
            reference=complex_entropy(cold)
            old=FE.entropy2(cold)
            C,checks=RE.real_correlation_matrix(state,indices)
            reconstruction_error=maximum(abs,C-real.(cold))
            eigenentropy=RE.entropy2_real!(copy(C);method=:eigen)
            cholentropy=RE.entropy2_real!(copy(C);method=:cholesky,covariance_validated=true)
            @assert reconstruction_error < 1e-12
            @assert max(abs(eigenentropy-reference),abs(cholentropy-reference),abs(old-reference)) < 1e-9
            push!(rows,(L=L,theta=theta,beta=1.0,reference_complex=reference,locked_FE=old,
                real_eigen=eigenentropy,real_cholesky=cholentropy,
                eigen_error=abs(eigenentropy-reference),cholesky_error=abs(cholentropy-reference),
                reconstruction_error=reconstruction_error,checks...))
            println("validated L=$L theta=$theta eigen_error=$(abs(eigenentropy-reference)) chol_error=$(abs(cholentropy-reference)) max_imag=$(checks.max_imaginary)")
            flush(stdout)
        end
    end
    csvwrite(joinpath(REAL_OUT,"real_entropy_validation.csv"),rows)
    return rows
end

function benchmark(L=90)
    p=only(filter(p->p.name=="random_half_3",experiment_patterns()))
    pre=FE.prepare_pattern(FE.BlochModel(L,3),p.occupied)
    state=FE.evolve(pre,16;beta=1)
    indices=FE.rectangle_indices(L)
    rows=NamedTuple[]
    GC.gc()
    reconstruction=@timed RE.real_correlation_matrix(state,indices)
    C,checks=reconstruction.value
    println("benchmark L=$L n=$(size(C,1)) real_reconstruction_seconds=$(reconstruction.time) max_imag=$(checks.max_imaginary)")
    flush(stdout)
    for method in (:cholesky,:eigen)
        GC.gc()
        Cwork=copy(C)
        result=@timed RE.entropy2_real!(Cwork;method,covariance_validated=true)
        push!(rows,(L=L,theta=16.0,beta=1.0,n=size(C,1),method=string(method),entropy=result.value,
            seconds=result.time,allocated_bytes=result.bytes,process_peak_rss_bytes=Sys.maxrss(),
            reconstruction_seconds=reconstruction.time,reconstruction_allocated_bytes=reconstruction.bytes,
            matrix_bytes=checks.matrix_bytes,max_imaginary=checks.max_imaginary,
            imaginary_frobenius=checks.imaginary_frobenius,real_part_entropy_bound=checks.real_part_entropy_bound))
        println("benchmark $(rows[end])"); flush(stdout)
        Cwork=nothing
    end
    GC.gc()
    complex_reconstruction=@timed FE.correlation_matrix(state,indices)
    cold=complex_reconstruction.value
    GC.gc()
    result=@timed complex_entropy(cold)
    push!(rows,(L=L,theta=16.0,beta=1.0,n=size(C,1),method="complex_eigen",entropy=result.value,
        seconds=result.time,allocated_bytes=result.bytes,process_peak_rss_bytes=Sys.maxrss(),
        reconstruction_seconds=complex_reconstruction.time,reconstruction_allocated_bytes=complex_reconstruction.bytes,
        matrix_bytes=sizeof(cold),max_imaginary=checks.max_imaginary,
        imaginary_frobenius=checks.imaginary_frobenius,real_part_entropy_bound=checks.real_part_entropy_bound))
    println("benchmark $(rows[end])"); flush(stdout)
    @assert maximum(r.entropy for r in rows)-minimum(r.entropy for r in rows)<1e-9
    csvwrite(joinpath(REAL_OUT,"real_entropy_benchmark.csv"),rows)
    return rows
end

function main()
    mkpath(REAL_OUT)
    mode=isempty(ARGS) ? "validate" : ARGS[1]
    rows=mode=="benchmark" ? benchmark(length(ARGS)>1 ? parse(Int,ARGS[2]) : 90) : validate()
    path=joinpath(REAL_OUT,"real_entropy_$(mode)_manifest.txt")
    open(path,"w") do io
        println(io,"date=$(Dates.now())\njulia=$VERSION\nblas_threads=4\nmode=$mode\nrows=$(length(rows))")
        println(io,"RSS is process cumulative high-water mark; allocated_bytes is allocation traffic, not peak live memory.")
        for name in ("fermion_experiment.jl","experiment_patterns.jl","real_entropy_extension.jl","real_entropy_checks.jl")
            println(io,"sha256_$name=$(bytes2hex(sha256(read(joinpath(@__DIR__,name)))))")
        end
    end
end
if abspath(PROGRAM_FILE)==@__FILE__
    main()
end
