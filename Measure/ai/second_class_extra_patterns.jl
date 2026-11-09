"""
Four additional class-2 patterns, selected by geometry and spectral rank only.

Bits are A(x,y),B(x,y), x fastest, then y, starting at (1,1).
All primitive cells are either empty or doubly occupied and filling is 1/2.

The staircase m=4 pattern has row lengths (4,3,1,0) of doubly occupied cells.
The notched stripe starts with columns x=1,2 full, then exchanges the full
cell (1,1) with the empty cell (3,2).  These are the first rank-accessible
examples in the recorded geometry-only candidate families. Neither is a
translation, complement, or lattice-symmetry copy of either old m=4 example:
occupied nearest-neighbour degree histograms provide an exact distinction.

The two m=2 alternating controls are known to be ground-inaccessible. They
must retain their strict accessible excited-state limit, not a false ground
state introduced by floating-point leakage during very long direct QR runs.
"""
function second_class_extra_patterns()
    entries = (
        ("pure02_staircase4", 4, "11111111111111001100000000000000", true),
        ("pure02_notched_stripe4", 4, "00110000111111001111000011110000", true),
        ("pure02_alternating_x", 2, "11001100", false),
        ("pure02_alternating_checker", 2, "11000011", false),
    )
    patterns = [(; name, family="general", m,
        occupied=BitVector(c=='1' for c in bits), class_id=2, expected_ground)
        for (name,m,bits,expected_ground) in entries]
    for p in patterns
        @assert length(p.occupied)==2p.m^2 && count(p.occupied)==p.m^2
        @assert all(p.occupied[2i-1]==p.occupied[2i] for i in 1:p.m^2)
    end
    return patterns
end
