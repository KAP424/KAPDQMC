"""
Four matched-statistics patterns selected by pattern_arrangement_controls.jl.
All four reach the global ground energy at L=9 and L=18 in the full census.
Special pair: A-B=5, doubles=empty=0. General pair: A-B=1, doubles=empty=1.
The pairs are distinct even after averaging every supercell translation.
Orbital order is A(x,y), B(x,y), x fastest, as in fermion_experiment.jl.
"""
function matched_arrangement_patterns()
    entries=(
        ("special_matched_a","special","100110011010101010"),
        ("special_matched_b","special","101001011010101010"),
        ("general_matched_a","general","011010111001011000"),
        ("general_matched_b","general","101001010110101100"),
    )
    return [(;name,family,m=3,occupied=BitVector(c=='1' for c in bits))
        for (name,family,bits) in entries]
end
