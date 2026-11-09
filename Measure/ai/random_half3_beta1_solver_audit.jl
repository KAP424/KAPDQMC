# Independent small-Bloch-block audit; no large real-space covariance or entropy.
# julia --startup-file=no --project=. src/Measure/ai/random_half3_beta1_solver_audit.jl
using LinearAlgebra, SHA, TOML, Dates
BLAS.set_num_threads(1)
include(joinpath(@__DIR__,"fermion_experiment.jl"))
import .FermionExperiment as FE
include(joinpath(@__DIR__,"experiment_patterns.jl"))

const AUDIT_DATA=joinpath(@__DIR__,"data")
const AUDIT_CSV=joinpath(AUDIT_DATA,"random_half3_beta1_solver_audit.csv")
const AUDIT_TOML=joinpath(AUDIT_DATA,"random_half3_beta1_solver_audit.toml")

function distances(a,b)
    total=0.0; maxblock=0.0; maxentry=0.0
    for (qa,qb) in zip(a.orbitals,b.orbitals)
        diff=qa*qa'-qb*qb'
        d=norm(diff)
        total+=d^2
        maxblock=max(maxblock,d)
        maxentry=max(maxentry,maximum(abs,diff))
    end
    return (normalized_projector_difference=sqrt(total)/a.prepared.model.L,
        max_block_projector_difference=maxblock,max_projector_entry_difference=maxentry)
end

function independent_measure(pre,orbitals,theta,beta)
    measured=Matrix{ComplexF64}[]
    for q in orbitals
        raw=copy(q)
        raw[pre.E,:] .*= exp(-2beta)
        push!(measured,Matrix(qr(raw).Q)[:,1:length(pre.F)])
    end
    return FE.FermionState(pre,theta,beta,measured)
end

function audit_row(kind,state,reference,limitdiag;tol=1e-10,dt=0.0,steps=0)
    pre=state.prepared
    d=FE.diagnostics(state)
    r=FE.diagnostics(reference)
    delta=distances(state,reference)
    rows_preserved=pre.F==findall(pre.occupied) && pre.E==findall(.!pre.occupied)
    @assert rows_preserved
    @assert abs(limitdiag.excess_energy)<1e-8
    @assert limitdiag.residual<1e-9
    @assert delta.normalized_projector_difference<1e-9
    @assert delta.max_block_projector_difference<1e-8
    return (check=kind,pattern="random_half_3",L=pre.model.L,theta=state.theta,beta=state.beta,
        tol=tol,reference_tol=1e-10,dt=dt,steps=steps,delta...,
        energy_difference=abs(d.energy-r.energy),energy=d.energy,excess_energy=d.excess_energy,
        residual=d.residual,premeasurement_limit_excess=limitdiag.excess_energy,
        premeasurement_limit_residual=limitdiag.residual,
        reconstruction_error=pre.max_reconstruction_error,min_pivot_sv=pre.min_pivot_sv,
        orthogonality_error=d.projector_error,original_occupied_rows_preserved=rows_preserved)
end

function record!(rows,io,row)
    isempty(rows) && println(io,join(keys(row),','))
    push!(rows,row)
    println(io,join(values(row),','))
    flush(io)
end

function stepped_checks!(rows,io,pattern)
    dt=0.25
    for L in (18,36)
        model=FE.BlochModel(L,3)
        pre=FE.prepare_pattern(model,pattern.occupied;tol=1e-10)
        limitdiag=FE.diagnostics(FE.evolve(pre,Inf))
        p=length(pre.F)
        # Matrix exponentials are formed directly from h, independently of the
        # energy-adapted coefficients, gaps and pivots used by FE.evolve.
        propagators=[exp(-dt*h) for h in model.hamiltonians]
        Q0=zeros(ComplexF64,2p,p)
        Q0[pre.F,:]=Matrix{ComplexF64}(I,p,p)
        qs=[copy(Q0) for _ in model.hamiltonians]
        for step in 1:256
            for k in eachindex(qs)
                qs[k]=Matrix(qr(propagators[k]*qs[k]).Q)[:,1:p]
            end
            theta=step*dt
            theta in (16.0,32.0,64.0) || continue
            reference=FE.evolve(pre,theta)
            for beta in (0.0,1.0)
                stepped=independent_measure(pre,qs,theta,beta)
                expected=beta==0 ? reference : FE.measure(reference,beta)
                row=audit_row("independent_step_QR",stepped,expected,limitdiag;dt,steps=step)
                record!(rows,io,row)
            end
            println("step QR complete L=$L theta=$theta max_block_error=$(rows[end].max_block_projector_difference)")
            flush(stdout)
        end
    end
end

function tolerance_checks!(rows,io,pattern)
    for L in (144,180,216)
        model=FE.BlochModel(L,3)
        refpre=FE.prepare_pattern(model,pattern.occupied;tol=1e-10)
        references=Dict((theta,beta)=>FE.evolve(refpre,theta;beta)
            for theta in (32.0,Inf) for beta in (0.0,1.0))
        for tol in (1e-8,1e-10,1e-12)
            pre=tol==1e-10 ? refpre : FE.prepare_pattern(model,pattern.occupied;tol)
            limitstate=FE.evolve(pre,Inf)
            limitdiag=FE.diagnostics(limitstate)
            finite=FE.evolve(pre,32.0)
            for theta in (32.0,Inf), beta in (0.0,1.0)
                raw=theta==Inf ? limitstate : finite
                state=beta==0 ? raw : FE.measure(raw,beta)
                row=audit_row("rank_tolerance",state,references[(theta,beta)],limitdiag;tol)
                record!(rows,io,row)
            end
            println("tolerance complete L=$L tol=$tol ground_excess=$(limitdiag.excess_energy) residual=$(limitdiag.residual)")
            flush(stdout)
        end
        GC.gc(false)
    end
end

function main()
    mkpath(AUDIT_DATA)
    pattern=only(filter(p->p.name=="random_half_3",experiment_patterns()))
    rows=NamedTuple[]
    started=time()
    open(AUDIT_CSV,"w") do io
        stepped_checks!(rows,io,pattern)
        tolerance_checks!(rows,io,pattern)
    end
    step_rows=filter(r->r.check=="independent_step_QR",rows)
    tol_rows=filter(r->r.check=="rank_tolerance",rows)
    summary=Dict(
        "utc"=>string(Dates.now(Dates.UTC)),"julia_version"=>string(VERSION),
        "blas_threads"=>BLAS.get_num_threads(),"pattern"=>pattern.name,
        "pattern_bits"=>join(Int.(pattern.occupied)),"rows"=>length(rows),
        "runtime_seconds"=>time()-started,"process_peak_rss_bytes"=>Int(Sys.maxrss()),
        "step_sizes"=>[18,36],"step_thetas"=>[16,32,64],"step_dt"=>0.25,
        "tolerance_sizes"=>[144,180,216],"tolerance_finite_theta"=>32,
        "rank_tolerances"=>[1e-8,1e-10,1e-12],"measurement_betas"=>[0,1],
        "distance_definition"=>"Frobenius norm of full projector difference divided by L=sqrt(Nparticles); also max individual 18x18 block and entry errors",
        "independent_algorithm"=>"exact exp(-dt*h), initial occupied site columns, thin QR every dt; measurement exp(-2beta) on original empty rows followed by an independent QR",
        "finite_theta_is_not_ground_limit"=>true,"large_region_entropies_calculated"=>false,
        "all_original_occupied_rows_preserved"=>all(r.original_occupied_rows_preserved for r in rows),
        "max_step_normalized_projector_difference"=>maximum(r.normalized_projector_difference for r in step_rows),
        "max_step_block_projector_difference"=>maximum(r.max_block_projector_difference for r in step_rows),
        "max_step_energy_difference"=>maximum(r.energy_difference for r in step_rows),
        "max_tolerance_normalized_projector_difference"=>maximum(r.normalized_projector_difference for r in tol_rows),
        "max_tolerance_block_projector_difference"=>maximum(r.max_block_projector_difference for r in tol_rows),
        "max_tolerance_energy_difference"=>maximum(r.energy_difference for r in tol_rows),
        "max_ground_limit_absolute_excess"=>maximum(abs(r.premeasurement_limit_excess) for r in rows),
        "max_ground_limit_residual"=>maximum(r.premeasurement_limit_residual for r in rows),
        "max_reconstruction_error"=>maximum(r.reconstruction_error for r in rows),
        "minimum_retained_pivot_sv"=>minimum(r.min_pivot_sv for r in rows),
        "max_orthogonality_error"=>maximum(r.orthogonality_error for r in rows))
    hashes=Dict(name=>bytes2hex(sha256(read(joinpath(@__DIR__,name)))) for name in
        ("fermion_experiment.jl","experiment_patterns.jl","random_half3_beta1_solver_audit.jl"))
    hashes[basename(AUDIT_CSV)]=bytes2hex(sha256(read(AUDIT_CSV)))
    summary["sha256"]=hashes
    open(AUDIT_TOML,"w") do io
        TOML.print(io,summary;sorted=true)
    end
    println(read(AUDIT_TOML,String))
end

if abspath(PROGRAM_FILE)==@__FILE__
    main()
end
