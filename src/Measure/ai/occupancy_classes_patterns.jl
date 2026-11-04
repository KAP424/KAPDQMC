"""
Six half-filled periodic product configurations classified by cell occupation.

class_id = 1: every primitive cell has one fermion.
class_id = 2: every primitive cell has either zero or two fermions.
class_id = 3: single cells coexist with both empty and doubly occupied cells.

Bits follow A(x,y), B(x,y), with x fastest and then y. The patterns are
defined independently of previous catalogues. The pure-0/2 period-four
examples were selected by a small-block reachability screen; full requested
size checks are recorded separately in data/occupancy_classes_reachability.csv.
"""
function occupancy_class_patterns()
    entries = (
        ("one_B_2",           "special", 2, "10101001",                         1),
        ("diagonal_B_3",      "special", 3, "011010100110101001",               1),
        ("pure02_block4",     "general", 4, "11111111110011001100110000000000", 2),
        ("pure02_irregular4", "general", 4, "11001111001100000011001111001100", 2),
        ("random_half_3",     "general", 3, "100110001011010110",               3),
        ("general_matched_a", "general", 3, "011010111001011000",               3),
    )
    patterns = [(; name, family, m, occupied=BitVector(c == '1' for c in bits),
                  class_id) for (name, family, m, bits, class_id) in entries]
    @assert length(unique(p.name for p in patterns)) == 6
    @assert all(count(p -> p.class_id == cls, patterns) == 2 for cls in 1:3)
    for p in patterns
        @assert length(p.occupied) == 2p.m^2
        @assert count(p.occupied) == p.m^2
        cell_n = [Int(p.occupied[2i-1]) + Int(p.occupied[2i]) for i in 1:p.m^2]
        n0, n1, n2 = count(==(0), cell_n), count(==(1), cell_n), count(==(2), cell_n)
        @assert n0 == n2
        if p.class_id == 1
            @assert (n0, n1, n2) == (0, p.m^2, 0)
        elseif p.class_id == 2
            @assert (n0, n1, n2) == (p.m^2÷2, 0, p.m^2÷2)
        else
            @assert n0 > 0 && n1 > 0 && n2 > 0
        end
    end
    return patterns
end
