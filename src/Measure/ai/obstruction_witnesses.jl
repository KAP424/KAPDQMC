#!/usr/bin/env julia
#=
Independent analytic checks of period-two HoneyComb120 ground-state obstructions.
Run: julia --startup-file=no --project=. src/Measure/ai/obstruction_witnesses.jl
Only writes new obstruction_* evidence; existing sources and data are read-only.
Coordinates are zero based; f(k)=1+exp(ikx)+exp(-iky), hopping +1.
=#
using LinearAlgebra, Printf, SHA, Dates
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "fermion_experiment.jl"))
import .FermionExperiment as FE
include(joinpath(@__DIR__, "experiment_patterns.jl"))

const OUT = joinpath(@__DIR__, "data")
const NAMES = ["stripe_AB_x", "stripe_AB_y", "checker_AB", "full_empty_x", "full_empty_checker"]
const SPECS = Dict(
    "stripe_AB_x" => (k=(0.0,pi), Q=(pi,0.0), epsB=1, eigen=true),
    "stripe_AB_y" => (k=(pi,0.0), Q=(0.0,pi), epsB=1, eigen=true),
    "checker_AB" => (k=(0.0,0.0), Q=(pi,pi), epsB=-1, eigen=false),
    "full_empty_x" => (k=(0.0,0.0), Q=(pi,0.0), epsB=-1, eigen=false),
    "full_empty_checker" => (k=(0.0,pi), Q=(pi,pi), epsB=-1, eigen=true))

f(k) = 1 + cis(k[1]) + cis(-k[2])
function negative_bloch(L, k)
    phase = f(k)/abs(f(k))
    v = zeros(ComplexF64, 2L^2)
    for y in 0:L-1, x in 0:L-1
        z = cis(k[1]*x+k[2]*y)/(sqrt(2)*L)
        v[1+2*(x+L*y)] = -phase*z
        v[2+2*(x+L*y)] = z
    end
    return v
end

function analytic_excess(name, L)
    c = [2abs(cospi(2n/L)) for n in 0:L÷2-1]
    if name in ("checker_AB", "full_empty_x")
        return 2sum(max.(c .- 1, 0))
    end
    return L + 2sum(max.(1 .- c, 0)) - 2Int(L % 4 == 0)
end

function negative_E_eigenstates(eig, F)
    count_null = 0
    i = 1
    while i <= length(eig.values) && eig.values[i] < -1e-9
        j = i
        while j < length(eig.values) && abs(eig.values[j+1]-eig.values[i]) < 1e-9
            j += 1
        end
        count_null += count(<(1e-9), svdvals(eig.vectors[F,i:j]))
        i = j + 1
    end
    return count_null
end

function writecsv(path, rows)
    open(path,"w") do io
        println(io,join(keys(first(rows)),','))
        for r in rows
            println(io,join(values(r),','))
        end
    end
end

function dense_witness_checks(patterns)
    rows = NamedTuple[]
    for L in (4,6,8,10,12)
        H = FE.dense_hamiltonian(L)
        eig = eigen(Hermitian(H))
        Vminus = eig.vectors[:,eig.values .< -1e-9]
        model = FE.BlochModel(L,2)
        for p in patterns
            spec = SPECS[p.name]
            pre = FE.prepare_pattern(model,p.occupied)
            F = FE.occupied_indices(pre)
            w = (negative_bloch(L,spec.k) + spec.epsB*negative_bloch(L,spec.k .+ spec.Q))/sqrt(2)
            meanenergy = real(dot(w,H*w))
            normerr = abs(norm(w)-1)
            Fleak = norm(w[F])
            negleak = norm(w-Vminus*(Vminus'*w))
            residual = norm(H*w-meanenergy*w)
            neigen = negative_E_eigenstates(eig,F)
            expected = spec.eigen ? L÷2 : 0
            @assert max(normerr,Fleak,negleak) < 1e-12
            @assert neigen == expected
            @assert abs(meanenergy - (spec.eigen ? -1 : -2)) < 1e-12
            @assert abs(residual - (spec.eigen ? 0 : 1)) < 1e-12
            push!(rows,(pattern=p.name,L=L,witness=spec.eigen ? "negative_eigenstate" : "negative_spectral_combination",
                norm_error=normerr,occupied_F_leakage=Fleak,nonnegative_spectral_leakage=negleak,
                mean_energy=meanenergy,eigen_residual_about_mean=residual,
                negative_eigenstates_supported_E=neigen,expected_negative_eigenstates_supported_E=expected))
        end
    end
    writecsv(joinpath(OUT,"obstruction_witness_checks.csv"),rows)
    return rows
end

function gamma_checks()
    # Cell order (0,0),(1,0),(0,1),(1,1); A orbitals first in this proof.
    model = FE.BlochModel(6,2)
    T = real(model.hamiltonians[1][1:2:8,2:2:8])
    s = [1.0,-1.0,-1.0,1.0]
    U = Matrix{Float64}(I,4,4)-s*s'/2
    ev = eigen(Symmetric(T))
    Uspectral = ev.vectors*Diagonal(sign.(ev.values))*ev.vectors'
    @assert norm(U-Uspectral) < 1e-12
    @assert norm(T-T') < 1e-12
    Vnegative = vcat(-U,Matrix{Float64}(I,4,4))/sqrt(2)
    rows = NamedTuple[]
    blockedmasks = Int[]
    for mask in 0:255
        count_ones(mask)==4 || continue
        occ = BitVector([!iszero(mask & (1<<i)) for i in 0:7])
        A = findall(occ[1:2:8]); B = findall(occ[2:2:8])
        cellcounts = Int.(occ[1:2:8]) + Int.(occ[2:2:8])
        specialbalanced = all(==(1),cellcounts) && length(A)==2
        twodouble = count(==(2),cellcounts)==2
        analyticblocked = specialbalanced || twodouble
        initialrows = vcat(A,4 .+ B)
        minsv = minimum(svdvals(Vnegative[initialrows,:]))
        pre = FE.prepare_pattern(model,occ)
        excess = pre.accessible_energy-model.ground_energy
        @assert (minsv<1e-10) == analyticblocked
        @assert (excess>1e-8) == analyticblocked
        subminor = specialbalanced ? det(U[A,A]) : twodouble ? det(U[A,setdiff(1:4,A)]) : NaN
        analyticblocked && push!(blockedmasks,mask)
        push!(rows,(mask=mask,occupied_bits=join(Int.(occ)),A_count=length(A),B_count=length(B),
            doubles=count(==(2),cellcounts),balanced_special=specialbalanced,two_full_two_empty=twodouble,
            gamma_min_overlap_sv=minsv,reduced_U_minor=subminor,analytic_obstruction=analyticblocked,
            L6_accessible_excess=excess))
    end
    @assert length(blockedmasks)==12
    # Verify the twelve analytic masks on even sizes not restricted to multiples of six.
    size_rows = NamedTuple[]
    for L in (4,6,8,10,12)
        model = FE.BlochModel(L,2)
        for mask in blockedmasks
            occ = BitVector([!iszero(mask & (1<<i)) for i in 0:7])
            pre = FE.prepare_pattern(model,occ)
            excess = pre.accessible_energy-model.ground_energy
            @assert excess > 1e-8
            push!(size_rows,(L=L,mask=mask,occupied_bits=join(Int.(occ)),accessible_excess=excess))
        end
    end
    writecsv(joinpath(OUT,"obstruction_gamma_masks.csv"),rows)
    writecsv(joinpath(OUT,"obstruction_gamma_sizes.csv"),size_rows)
    return norm(U-Uspectral),rows,size_rows
end

function excess_checks(patterns)
    rows = NamedTuple[]
    for L in vcat(collect(4:2:24),[30,36])
        model = FE.BlochModel(L,2)
        for p in patterns
            pre = FE.prepare_pattern(model,p.occupied)
            value = pre.accessible_energy-model.ground_energy
            formula = analytic_excess(p.name,L)
            error = abs(formula-value)
            @assert error < 1e-9
            push!(rows,(pattern=p.name,L=L,analytic_excess=formula,numeric_excess=value,absolute_error=error,
                source="independent_Bloch_prepare"))
        end
    end
    # Read prior large-L results; no new large-L calculation and no data rewriting.
    lines = readlines(joinpath(OUT,"equilibrium_extended.csv"))
    columns = split(first(lines),',')
    idx = Dict(x=>i for (i,x) in enumerate(columns))
    for line in lines[2:end]
        cells = split(line,',')
        name = cells[idx["pattern"]]
        name in NAMES || continue
        parse(Float64,cells[idx["beta"]]) == 0 || continue
        L = parse(Int,cells[idx["L"]])
        value = parse(Float64,cells[idx["premeasurement_excess"]])
        formula = analytic_excess(name,L)
        error = abs(formula-value)
        @assert error < 1e-8
        push!(rows,(pattern=name,L=L,analytic_excess=formula,numeric_excess=value,absolute_error=error,
            source="existing_equilibrium_extended"))
    end
    writecsv(joinpath(OUT,"obstruction_exact_excess.csv"),rows)
    return rows
end

function main()
    mkpath(OUT)
    started = time()
    patterns = filter(p->p.name in NAMES,experiment_patterns())
    witnesses = dense_witness_checks(patterns)
    gammaerr,masks,sizes = gamma_checks()
    excess = excess_checks(patterns)
    summary = [
        "date=$(Dates.now())", "julia=$VERSION", "blas_threads=1",
        "coordinates=zero based; hopping=+1; A(x,y) links B(x,y), B(x+1,y), B(x,y-1)",
        "f(k)=1+exp(i*kx)+exp(-i*ky)",
        "applicability=all even L>=4; L>=4 avoids duplicate-neighbor conventions at L=2",
        "witness_rows=$(length(witnesses)); gamma_masks=$(length(masks)); blocked_gamma_masks=12; blocked_size_rows=$(length(sizes))",
        "gamma_polar_U_error=$gammaerr",
        "max_witness_norm_error=$(maximum(r.norm_error for r in witnesses))",
        "max_witness_F_leakage=$(maximum(r.occupied_F_leakage for r in witnesses))",
        "max_witness_nonnegative_spectral_leakage=$(maximum(r.nonnegative_spectral_leakage for r in witnesses))",
        "excess_rows=$(length(excess)); max_exact_excess_error=$(maximum(r.absolute_error for r in excess))",
        "seconds=$(time()-started)"]
    for name in ("obstruction_witnesses.jl","fermion_experiment.jl","experiment_patterns.jl")
        push!(summary,"sha256_$name=$(bytes2hex(sha256(read(joinpath(@__DIR__,name)))))")
    end
    for name in ("obstruction_witness_checks.csv","obstruction_gamma_masks.csv","obstruction_gamma_sizes.csv",
                 "obstruction_exact_excess.csv","equilibrium_extended.csv")
        push!(summary,"sha256_$name=$(bytes2hex(sha256(read(joinpath(OUT,name)))))")
    end
    text=join(summary,'\n')*"\n"
    write(joinpath(OUT,"obstruction_checks_manifest.txt"),text)
    println(text)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
