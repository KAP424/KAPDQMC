using Random

"""Occupancy order: A(1,1), B(1,1), A(2,1), B(2,1), ... ."""
function special_pattern(m, b_cells)
    occupied = falses(2m^2)
    for y in 1:m, x in 1:m
        sublattice = (x, y) in b_cells ? 2 : 1
        occupied[sublattice + 2 * ((x - 1) + m * (y - 1))] = true
    end
    return occupied
end

function general_pattern(m, filled_cells, single_cells=Tuple{Int,Int,Int}[])
    occupied = falses(2m^2)
    for (x, y) in filled_cells, sublattice in 1:2
        occupied[sublattice + 2 * ((x - 1) + m * (y - 1))] = true
    end
    for (x, y, sublattice) in single_cells
        occupied[sublattice + 2 * ((x - 1) + m * (y - 1))] = true
    end
    @assert count(occupied) == m^2
    return occupied
end

function experiment_patterns()
    patterns = NamedTuple[]
    add(name, family, m, occupied) = push!(patterns, (; name, family, m, occupied))
    add("uniform_A", "special", 1, Bool[1, 0])
    add("uniform_B", "special", 1, Bool[0, 1])
    add("stripe_AB_x", "special", 2, special_pattern(2, [(2, 1), (2, 2)]))
    add("stripe_AB_y", "special", 2, special_pattern(2, [(1, 2), (2, 2)]))
    add("checker_AB", "special", 2, special_pattern(2, [(1, 1), (2, 2)]))
    add("one_B_2", "special", 2, special_pattern(2, [(2, 2)]))
    add("full_empty_x", "general", 2, general_pattern(2, [(1, 1), (1, 2)]))
    add("full_empty_checker", "general", 2, general_pattern(2, [(1, 1), (2, 2)]))
    add("one_B_3", "special", 3, special_pattern(3, [(3, 3)]))
    add("two_B_3", "special", 3, special_pattern(3, [(1, 1), (3, 3)]))
    add("diagonal_B_3", "special", 3, special_pattern(3, [(1, 1), (2, 2), (3, 3)]))
    add("full_empty_cluster3", "general", 3,
        general_pattern(3, [(1, 1), (2, 1), (3, 1), (1, 2)], [(2, 2, 1)]))
    random3 = falses(18)
    random3[randperm(MersenneTwister(424), 18)[1:9]] .= true
    add("random_half_3", "general", 3, random3)
    for p in patterns
        @assert length(p.occupied) == 2p.m^2 && count(p.occupied) == p.m^2
        singles = all(sum(p.occupied[2j-1:2j]) == 1 for j in 1:p.m^2)
        @assert singles == (p.family == "special")
    end
    return patterns
end

function write_pattern_catalogue(path)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "pattern,family,m,occupied_bits,double_cells,empty_cells,single_cells")
        for p in experiment_patterns()
            cells = [sum(p.occupied[2j-1:2j]) for j in 1:p.m^2]
            println(io, join((p.name, p.family, p.m, join(Int.(p.occupied)),
                count(==(2), cells), count(==(0), cells), count(==(1), cells)), ','))
        end
    end
end
