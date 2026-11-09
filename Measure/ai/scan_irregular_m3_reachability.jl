# Deterministic spatial candidate order; no entropy is computed or consulted.
# A 3x3 half-filled cell must contain 9 particles, so a pure 0/2 pattern
# cannot be half-filled. This screen uses four double, four empty, one A cell.
using LinearAlgebra, Printf, SHA
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "fermion_experiment.jl"))
using .FermionExperiment

function candidate_cells()
    # Cell index = x + 3(y-1). The sole single-occupied cell is A(3,3).
    candidates = [[1,2,4,5], [1,3,5,8], [1,2,6,8]]
    for a in 1:5, b in a+1:6, c in b+1:7, d in c+1:8
        cells = [a,b,c,d]
        cells in candidates || push!(candidates,cells)
    end
    @assert length(candidates) == 70
    return candidates
end

function candidate_occupation(cells)
    occ = falses(18)
    for i in cells
        occ[2i-1:2i] .= true
    end
    occ[17] = true
    @assert count(occ) == 9
    return occ
end

sizes = [18,27,36,45,54,63,72,81,90,108,126,144,180]
models = Dict(3 => BlochModel(3,3))
out = joinpath(@__DIR__, "data", "irregular_m3_candidate_screen.csv")
selected = Ref("")
selected_rank = Ref(0)
open(out, "w") do io
    println(io,"candidate_order,occupied_bits,double_cells,single_cell,L,excess_energy,ground_reachable,min_pivot_sv,reconstruction_error")
    for (i,cells) in enumerate(candidate_cells())
        occ = candidate_occupation(cells)
        bits = join(Int.(occ))
        good = true
        for L in vcat(3,sizes)
            model = get!(models,L) do
                BlochModel(L,3)
            end
            pre = prepare_pattern(model,occ)
            excess = pre.accessible_energy-model.ground_energy
            pass = abs(excess) < 1e-8
            println(io,join((i,bits,join(cells,';'),9,L,excess,pass,
                pre.min_pivot_sv,pre.max_reconstruction_error),','))
            flush(io)
            @printf("candidate=%d bits=%s L=%d ground=%s excess=%.4g min_sv=%.4g\n",
                    i,bits,L,pass,excess,pre.min_pivot_sv)
            flush(stdout)
            if !pass
                good = false
                break
            end
        end
        if good
            selected[] = bits
            selected_rank[] = i
            break
        end
    end
end
isempty(selected[]) && error("No candidate reachable on every requested size")
println("SELECTED candidate=",selected_rank[]," bits=",selected[])
