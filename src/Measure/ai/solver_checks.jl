using Test, LinearAlgebra, Random, Printf
include("fermion_experiment.jl")
using .FermionExperiment
include(joinpath(@__DIR__, "..", "..", "public", "Geometry.jl"))
include(joinpath(@__DIR__, "..", "Force2", "NeqTgrd.jl"))
BLAS.set_num_threads(parse(Int,get(ENV,"NEQ_BLAS_THREADS","1")))

function dense_state(pre, theta, beta=0.0)
    L = pre.model.L
    h = dense_hamiltonian(L)
    occupied = occupied_indices(pre)
    p0 = Matrix{ComplexF64}(I,2L^2,2L^2)[:,occupied]
    raw = exp(-theta*h)*p0
    empty = setdiff(1:2L^2,occupied)
    raw[empty,:] .*= exp(-2beta)
    return Matrix(qr(raw).Q)[:,1:L^2]
end

@testset "HoneyComb120 periodic free fermion solver" begin
    rng = MersenneTwister(4271)
    for m in (1,2,3)
        L = 6
        model = BlochModel(L,m)
        h = dense_hamiltonian(L)
        @test h == nnK_Matrix("HoneyComb120",[L,L])
        @test sort(vcat(model.energies...)) ≈ eigvals(Hermitian(h)) atol=2e-13
        special = special_pattern([isodd(x+y) ? 1 : -1 for x in 1:m,y in 1:m])
        general = falses(2m^2)
        general[randperm(rng,2m^2)[1:m^2]] .= true
        for pattern in (special,general)
            pre = prepare_pattern(model,pattern)
            occ = occupied_indices(pre)
            @test length(occ) == L^2
            @test sort(vcat(occ,setdiff(1:2L^2,occ))) == collect(1:2L^2)
            @test pre.max_reconstruction_error < 1e-11
            @test entropy_strip(evolve(pre,0.0)) < 1e-11
            @test entropy_rectangle(evolve(pre,0.0)) < 1e-11
            for theta in (0.0,0.01,0.2,1.0,3.0), beta in (0.0,0.4,3.0)
                state = evolve(pre,theta;beta=beta)
                c = correlation_matrix(state)
                reference = dense_state(pre,theta,beta)
                @test norm(c-reference*reference') < 2e-8
                @test norm(c-c') < 1e-12
                @test norm(c*c-c) < 1e-11
                @test real(tr(c)) ≈ L^2 atol=1e-11
                @test entropy_strip(state) ≈ entropy2(c[rectangle_indices(L;height=L),rectangle_indices(L;height=L)]) atol=2e-11
                @test entropy_strip(state;orientation=:y) ≈ entropy2(c[rectangle_indices(L;width=L,height=L÷3),rectangle_indices(L;width=L,height=L÷3)]) atol=2e-11
                @test entropy_rectangle(state) ≈ entropy2(c[rectangle_indices(L),rectangle_indices(L)]) atol=2e-11
            end
            # Imaginary-time derivative dC/dtheta=-hC-Ch+2ChC.
            c0 = correlation_matrix(evolve(pre,0.0))
            dt = 1e-6
            derivative = (correlation_matrix(evolve(pre,dt))-c0)/dt
            @test norm(derivative-(-h*c0-c0*h+2c0*h*c0)) < 8e-5
            equilibrium = evolve(pre,Inf)
            cq = correlation_matrix(equilibrium)
            @test diagnostics(equilibrium).residual < 1e-10
            @test projector_distance(equilibrium,evolve(pre,1000)) < 1e-10
            @test entropy_strip(measure(equilibrium,12.0)) < 1e-9
            @test entropy_rectangle(measure(equilibrium,12.0)) < 1e-9
            direct = Matrix{ComplexF64}(I,2L^2,2L^2)[:,occ]
            eig = eigen(Hermitian(h))
            NeqTgrd!(direct,exp(-0.1h);spectrum=(eig.values,eig.vectors))
            @test norm(cq-direct*direct') < 1e-9
            @test diagnostics(equilibrium).energy ≈ real(tr(h*cq)) atol=1e-10
            @test diagnostics(equilibrium).energy ≈ pre.accessible_energy atol=1e-10
        end
    end
    # Strong beta acts additively and never changes particle number.
    pre = prepare_pattern(BlochModel(6,2),Bool[1,1,0,0,1,0,0,1])
    a = measure(measure(evolve(pre,0.75),0.2),0.7)
    b = evolve(pre,0.75;beta=0.9)
    @test projector_distance(a,b) < 1e-12
    @test_throws ArgumentError prepare_pattern(BlochModel(6,2),ones(Int,8))
    @test_throws ArgumentError evolve(pre,-0.1)
    @test_throws ArgumentError measure(evolve(pre,1.0),-1.0)
end

println("Timing pilot ($(BLAS.get_num_threads()) BLAS threads; seconds; includes both strip directions + rectangle)")
for L in (isempty(ARGS) ? [18,36] : parse.(Int,ARGS))
    model_time = @elapsed model = BlochModel(L,3)
    prep_time = @elapsed pre = prepare_pattern(model,special_pattern([1 1 1; 1 1 1; 1 1 -1]))
    evolve_time = @elapsed state = evolve(pre,Inf;beta=0.5)
    strip_time = @elapsed sa = entropy_strip(state)
    rotated_time = @elapsed sc = entropy_strip(state;orientation=:y)
    rectangle_time = @elapsed sb = entropy_rectangle(state)
    @printf("L=%d model=%.4f prep=%.4f evolve=%.4f strip=%.4f rotated=%.4f rectangle=%.4f SA=%.12f SB=%.12f SC=%.12f\n",
        L,model_time,prep_time,evolve_time,strip_time,rotated_time,rectangle_time,sa,sb,sc)
end
