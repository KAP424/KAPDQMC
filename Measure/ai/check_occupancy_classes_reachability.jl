# Check spectral reachability only, without constructing real-space covariance
# matrices or calculating entropy. Run with Julia's standard LinearAlgebra.
using LinearAlgebra, Printf, SHA
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "fermion_experiment.jl"))
include(joinpath(@__DIR__, "occupancy_classes_patterns.jl"))
using .FermionExperiment

patterns = occupancy_class_patterns()
sizes = vcat(collect(24:12:144), [180, 216])
out = joinpath(@__DIR__, "data", "occupancy_classes_reachability.csv")
pattern_hash = bytes2hex(sha256(read(joinpath(@__DIR__, "occupancy_classes_patterns.jl"))))
kernel_hash = bytes2hex(sha256(read(joinpath(@__DIR__, "fermion_experiment.jl"))))
rank_tol, ground_tol = 1e-10, 1e-8
failures = Ref(0)
open(out, "w") do io
    println(io, "pattern,family,class_id,m,L,occupied_bits,empty_cells,single_cells,double_cells,ground_energy,accessible_energy,excess_energy,ground_reachable,min_pivot_sv,reconstruction_error,rank_tol,ground_tol,pattern_sha256,kernel_sha256")
    for m in sort(unique(p.m for p in patterns)), L in sizes
        L % m == 0 || continue
        model = BlochModel(L, m)
        for p in filter(p -> p.m == m, patterns)
            pre = prepare_pattern(model, p.occupied; tol=rank_tol)
            excess = pre.accessible_energy - model.ground_energy
            ground = abs(excess) < ground_tol
            failures[] += !ground
            st = pattern_statistics(p.occupied)
            n1 = m^2 - st.empty_cells - st.double_cells
            println(io, join((p.name,p.family,p.class_id,m,L,join(Int.(p.occupied)),
                st.empty_cells,n1,st.double_cells,model.ground_energy,pre.accessible_energy,
                excess,ground,pre.min_pivot_sv,pre.max_reconstruction_error,rank_tol,
                ground_tol,pattern_hash,kernel_hash), ','))
            flush(io)
            @printf("%s L=%d excess=%.4g ground=%s min_pivot=%.4g\n",
                    p.name,L,excess,ground,pre.min_pivot_sv)
            flush(stdout)
        end
        GC.gc()
    end
end
println("FINISH failures=", failures[], " output=", out)
