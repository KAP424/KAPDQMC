# Direct projector-rank audit plus an independent energy-accessibility check.
# No correlation matrix, entropy, or fitted result is used for selection.
using LinearAlgebra, Printf, SHA
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "fermion_experiment.jl"))
include(joinpath(@__DIR__, "irregular_m3_patterns.jl"))
using .FermionExperiment

sizes = [18,27,36,45,54,63,72,81,90,108,126,144,180]
rank_tol, zero_tol, ground_tol = 1e-10, 1e-9, 1e-8
pattern_file = joinpath(@__DIR__, "irregular_m3_patterns.jl")
kernel_file = joinpath(@__DIR__, "fermion_experiment.jl")
pattern_hash = bytes2hex(sha256(read(pattern_file)))
kernel_hash = bytes2hex(sha256(read(kernel_file)))
out = joinpath(@__DIR__, "data", "irregular_m3_reachability.csv")
open(out,"w") do io
    println(io,"pattern,m,L,occupied_bits,empty_cells,single_cells,double_cells,NA,NB,blocks,negative_rank_deficiency,nonpositive_rank_deficiency,min_negative_sv,min_nonpositive_sv,ground_energy,accessible_energy,excess_energy,ground_reachable,min_pivot_sv,reconstruction_error,rank_tol,zero_tol,ground_tol,pattern_sha256,kernel_sha256")
    for p in irregular_m3_patterns(), L in sizes
        model = BlochModel(L,p.m)
        F = findall(p.occupied)
        neg_deficit, nonpos_deficit = 0, 0
        min_neg_sv, min_nonpos_sv = Inf, Inf
        for k in eachindex(model.energies)
            e,v = model.energies[k], model.vectors[k]
            neg = e .< -zero_tol
            nonpos = e .<= zero_tol
            sigma_neg = svdvals(v[F,neg]')
            sigma_nonpos = svdvals(v[F,nonpos]')
            neg_deficit = max(neg_deficit,count(neg)-count(>(rank_tol),sigma_neg))
            nonpos_deficit = max(nonpos_deficit,p.m^2-count(>(rank_tol),sigma_nonpos))
            min_neg_sv = min(min_neg_sv,minimum(sigma_neg))
            min_nonpos_sv = min(min_nonpos_sv,minimum(sigma_nonpos))
        end
        pre = prepare_pattern(model,p.occupied;tol=rank_tol)
        excess = pre.accessible_energy-model.ground_energy
        reachable = neg_deficit == 0 && nonpos_deficit == 0 && abs(excess) < ground_tol
        @assert reachable
        @assert pre.max_reconstruction_error < 1e-10
        println(io,join((p.name,p.m,L,join(Int.(p.occupied)),4,1,4,
            count(p.occupied[1:2:end]),count(p.occupied[2:2:end]),length(model.energies),
            neg_deficit,nonpos_deficit,min_neg_sv,min_nonpos_sv,
            model.ground_energy,pre.accessible_energy,excess,reachable,
            pre.min_pivot_sv,pre.max_reconstruction_error,rank_tol,zero_tol,
            ground_tol,pattern_hash,kernel_hash),','))
        flush(io)
        @printf("%s L=%d rank=(%d,%d) min_sigma=(%.5g,%.5g) excess=%.4g reachable=%s\n",
            p.name,L,neg_deficit,nonpos_deficit,min_neg_sv,min_nonpos_sv,excess,reachable)
        flush(stdout)
        GC.gc()
    end
end
println("FINISH output=",out)
