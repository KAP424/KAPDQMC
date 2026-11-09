#=
Independent geometry and beta=0 checks for two balanced corner subtractions.

Run with `julia --project=. src/Measure/born_ai/check_born_boundary_geometry.jl`.
The full-translation identity applies to a sum of shared local cut-bond
densities only. It is not an identity for the complete entanglement entropy,
and does not remove corner, interaction-between-boundaries or nonlocal terms.
=#
using LinearAlgebra, Random, SHA, Dates, Printf, DelimitedFiles
include(joinpath(@__DIR__, "original_swap_scaling.jl"))
using .OriginalSwapScaling
BLAS.set_num_threads(1)

const OUTDIR = joinpath(@__DIR__, "data", "born_scaling", length(ARGS)>=2 ? ARGS[2] : "boundary_check")
const SIZES = isempty(ARGS) ? [6,12,18,24,30,36,42,48] : parse.(Int, split(ARGS[1], ','))
const MU = 1e-8
const SEED = 284102026

json(x::Nothing) = "null"
json(x::Bool) = x ? "true" : "false"
json(x::Number) = isfinite(x) ? string(x) : error("nonfinite JSON number")
json(x::AbstractString) = "\"" * replace(x, "\\"=>"\\\\", "\""=>"\\\"", "\n"=>"\\n", "\r"=>"\\r") * "\""
json(x::AbstractVector) = "[" * join(json.(x), ",") * "]"
json(x::Tuple) = json(collect(x))
json(x::AbstractDict) = "{" * join((json(string(k)) * ":" * json(v) for (k,v) in sort(collect(x);by=x->string(x[1]))), ",") * "}"
hashfile(path) = bytes2hex(sha256(read(path)))

function rect_indices(L, nx, ny, tx=0, ty=0)
    result = Int[]
    for y in 1:ny, x in 1:nx
        cell = mod1(x+tx,L) + (mod1(y+ty,L)-1)*L
        push!(result,2cell-1,2cell)
    end
    sort!(result)
end

function rect_regions(L, pn, pd, tx=0, ty=0)
    L % pd == 0 || error("nonintegral cut")
    w = L*pn÷pd
    (A=rect_indices(L,w,L,tx,ty), B=rect_indices(L,w,L-w,tx,ty),
     C=rect_indices(L,L,w,tx,ty))
end

function cut_indicators(L,nx,ny,tx,ty)
    # Coordinates label the A endpoint of each of the three honeycomb bonds.
    # The last index distinguishes A-inside/B-outside from its reverse, so
    # opposite microscopic terminations are never silently equated.
    inside = falses(L,L)
    for y in 1:ny, x in 1:nx
        inside[mod1(x+tx,L),mod1(y+ty,L)] = true
    end
    cuts = zeros(Int,L,L,3,2)
    for y in 1:L, x in 1:L, kind in 1:3
        bx,by = kind==1 ? (x,y) : kind==2 ? (mod1(x+1,L),y) : (x,mod1(y-1,L))
        a,b = inside[x,y],inside[bx,by]
        a != b && (cuts[x,y,kind,a ? 1 : 2] = 1)
    end
    cuts
end

function boundary_audit(L,pn,pd,rng)
    w=L*pn÷pd
    total=zeros(Int,L,L,3,2)
    subgroup=zeros(Int,L,L,3,2)
    fixed=zeros(Int,L,L,3,2)
    density=rand(rng,L,L,3,2) # Arbitrary positive, position- and type-dependent.
    directly_averaged=0.
    subgroup_directly_averaged=0.
    for ty in 0:L-1,tx in 0:L-1
        a=cut_indicators(L,w,L,tx,ty)
        b=cut_indicators(L,w,L-w,tx,ty)
        c=cut_indicators(L,L,w,tx,ty)
        weighted=(pd-pn).*a.+pn.*c.-pd.*b
        total .+= weighted
        directly_averaged+=sum(weighted.*density)/(pd*L^2)
        if tx%w==0 && ty%w==0
            subgroup .+= weighted
            subgroup_directly_averaged+=sum(weighted.*density)/(pd*pd^2)
        end
        if tx==0 && ty==0; fixed.=weighted; end
    end
    # Integer check is exact, without a floating tolerance.
    all(iszero,total) || error("translated local cut weights fail at L=$L p=$pn/$pd")
    all(iszero,subgroup) || error("subgroup local cut weights fail at L=$L p=$pn/$pd")
    pertype=[maximum(abs,view(total,:,:,kind,orientation)) for kind in 1:3 for orientation in 1:2]
    abs(directly_averaged)<1e-10 || error("floating density contraction unexpectedly inaccurate")
    abs(subgroup_directly_averaged)<1e-10 || error("subgroup floating density contraction unexpectedly inaccurate")
    fixed_density=sum(fixed.*density)/pd
    abs(fixed_density)>1e-8 || error("random illustration accidentally cancels")
    # The spatially constant coefficient cancels even at a fixed origin.
    fixed_type_sums=[sum(view(fixed,:,:,kind,orientation)) for kind in 1:3 for orientation in 1:2]
    all(iszero,fixed_type_sums) || error("fixed-origin direction/termination lengths do not balance")
    Dict("L"=>L,"p_num"=>pn,"p_den"=>pd,"translations"=>L^2,
        "integer_weight_denominator"=>pd,
        "subgroup_translations"=>pd^2,
        "subgroup_translation_coordinates"=>[(tx,ty) for tx in 0:w:L-1 for ty in 0:w:L-1],
        "max_absolute_subgroup_integer_weight"=>maximum(abs,subgroup),
        "max_absolute_subgroup_integer_weight_by_bond_then_orientation"=>[maximum(abs,view(subgroup,:,:,kind,orientation)) for kind in 1:3 for orientation in 1:2],
        "subgroup_random_density_D_direct_float"=>subgroup_directly_averaged,
        "subgroup_random_density_D_exact_integer_contraction"=>sum(subgroup.*density)/(pd*pd^2),
        "max_absolute_full_translation_integer_weight"=>maximum(abs,total),
        "max_absolute_integer_weight_by_bond_then_orientation"=>pertype,
        "fixed_origin_total_weight_by_bond_then_orientation"=>fixed_type_sums,
        "fixed_origin_random_density_D"=>fixed_density,
        "full_translation_random_density_D_direct_float"=>directly_averaged,
        "full_translation_random_density_D_exact_integer_contraction"=>sum(total.*density)/(pd*L^2))
end

function s2_eigen(G,region)
    eig=eigvals(Symmetric(G[region,region]))
    minimum(eig)>-1e-10 && maximum(eig)<1+1e-10 || error("unphysical spectrum")
    eig=clamp.(eig,0.,1.)
    -sum(log.(eig.^2 .+ (1 .-eig).^2))
end

function s2_cholesky(G,region)
    C=G[region,region]
    B=2C*C-2C+I
    -2sum(log,diag(cholesky(Symmetric(B)).U))
end

function read_legacy()
    paths=[joinpath(@__DIR__,"analysis","born_scaling_state_audit.csv"),
           joinpath(@__DIR__,"data","born_scaling","beta0_mass_large_sizes.csv")]
    values=Dict{Int,Float64}()
    for path in paths
        data,header=readdlm(path,',',Any,'\n';header=true)
        names=String.(vec(header)); lookup=Dict(n=>i for (i,n) in enumerate(names))
        for row in eachrow(data)
            haskey(lookup,"state") && row[lookup["state"]]!="staggered_mu1e-8" && continue
            values[Int(row[lookup["L"]])]=Float64(row[lookup["D"]])
        end
    end
    values,paths
end

function main()
    mkpath(OUTDIR)
    csvpath=joinpath(OUTDIR,"geometry_baseline.csv")
    auditpath=joinpath(OUTDIR,"boundary_audit.json")
    (isfile(csvpath)||isfile(auditpath)) && error("Refuse to overwrite existing boundary-check outputs")
    legacy,legacy_paths=read_legacy()
    rng=MersenneTwister(SEED)
    audits=Any[]; checks=Any[]
    start=now(UTC)
    open(csvpath,"w") do io
        println(io,"L,beta,mu,p_num,p_den,p,S_A,S_B,S_C,D,SA_minus_SC,D_cholesky,max_entropy_method_difference,legacy_D,legacy_D_difference,model_orthogonality_error,model_energy_error,model_residual,elapsed_seconds")
        for L in SIZES
            L%6==0 || error("use common multiples of 6")
            elapsed=@elapsed begin
                model=ground_model(L;state=:staggered,mu=MU)
                # independent indexing must reproduce the original region sets
                old=region_indices(L); new=rect_regions(L,1,3)
                all(sort(old[k])==new[k] for k in 1:3) || error("index convention mismatch")
                small_original = L<=12 ? entropy!(EntropyWorkspace(model),model,collect(1:model.n),0.) : nothing
                for (pn,pd) in ((1,3),(1,2))
                    bt=@elapsed audit=boundary_audit(L,pn,pd,rng)
                    push!(audits,audit)
                    regions=rect_regions(L,pn,pd)
                    et=@elapsed begin
                        ent=[s2_eigen(model.G,r) for r in regions]
                        chol=[s2_cholesky(model.G,r) for r in regions]
                    end
                    p=pn/pd; D=(1-p)*ent[1]+p*ent[3]-ent[2]
                    Dc=(1-p)*chol[1]+p*chol[3]-chol[2]
                    method_error=maximum(abs.(ent.-chol))
                    method_error<1e-8 || error("entropy algorithms disagree")
                    oldD=pn==1 && pd==3 && haskey(legacy,L) ? legacy[L] : nothing
                    difference=oldD===nothing ? nothing : D-oldD
                    difference===nothing || abs(difference)<1e-8 || error("legacy baseline mismatch")
                    if small_original!==nothing && pd==3
                        abs(D-small_original.D)<1e-9 || error("production entropy mismatch")
                    end
                    if L==6
                        # On a homogeneous beta=0 state every translated entropy
                        # must agree, not merely the local-boundary surrogate.
                        translated=[(1-p)*s2_eigen(model.G,rs.A)+p*s2_eigen(model.G,rs.C)-s2_eigen(model.G,rs.B)
                            for tx in 0:L-1 for ty in 0:L-1 for rs in (rect_regions(L,pn,pd,tx,ty),)]
                        err=maximum(abs.(translated.-D))
                        err<1e-10 || error("beta=0 translation invariance failed")
                        push!(checks,Dict("L"=>L,"p_num"=>pn,"p_den"=>pd,"full_entropy_translation_max_difference"=>err))
                    end
                    row=(L,0.,MU,pn,pd,p,ent[1],ent[2],ent[3],D,ent[1]-ent[3],Dc,method_error,
                        oldD===nothing ? "" : oldD,difference===nothing ? "" : difference,
                        model.orthogonality_error,model.ground_energy_error,model.eigenvector_residual,bt+et)
                    println(io,join(row,','));flush(io)
                    @printf("L=%d p=%d/%d D=%.14f eig/chol=%.3g boundary=%.2fs entropy=%.2fs\n",L,pn,pd,D,method_error,bt,et)
                    flush(stdout)
                end
            end
            @printf("finished L=%d total %.2fs\n",L,elapsed);flush(stdout)
            GC.gc()
        end
    end
    sources=[@__FILE__,joinpath(@__DIR__,"original_swap_scaling.jl"),joinpath(@__DIR__,"..","..","public","Geometry.jl")]
    receipt=Dict("status"=>"PASS","created_utc"=>string(now(UTC)),"started_utc"=>string(start),
        "sizes"=>SIZES,"beta"=>0.,"mu"=>MU,"BLAS_threads"=>BLAS.get_num_threads(),"random_density_seed"=>SEED,
        "boundary_checks"=>audits,"translation_entropy_checks"=>checks,
        "scope"=>"Exact integer identity for shared local cut-bond density, with distinct bond types and A-inside/B-inside terminations. This is not an identity for complete entropy; corner and nonlocal terms are not represented by the surrogate.",
        "baseline_method"=>"Ground projector uses existing staggered SVD model. Regional S2 from eigenvalues, independently checked against purity Cholesky for every size and against original entropy! for L<=12.",
        "source_hashes"=>Dict(replace(relpath(p,@__DIR__),'\\'=>'/')=>hashfile(p) for p in sources),
        "reference_hashes"=>Dict(replace(relpath(p,@__DIR__),'\\'=>'/')=>hashfile(p) for p in legacy_paths),
        "files"=>Dict("geometry_baseline.csv"=>hashfile(csvpath)))
    open(io->println(io,json(receipt)),auditpath,"w")
    println("PASS: ",auditpath)
end

abspath(PROGRAM_FILE)==abspath(@__FILE__) && main()
