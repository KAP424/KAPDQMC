"""
Half-filled 3x3 control closest to the pure-0/2 occupation class.

There are four empty cells, one A-only cell, and four double cells, so
N=9. An exactly pure-0/2 3x3 product pattern cannot have odd N=9.
Bit order: A(x,y), B(x,y), x fastest, then y. Cell occupations by row
(y=1 first) are [2 2 0; 0 0 2; 0 2 1], with the single cell on A.

Selected as the first GS-accessible member of a deterministic spatial
candidate order by scan_irregular_m3_reachability.jl. No entropy or fitted
coefficient was used for selection. See data/irregular_m3_candidate_screen.csv.
This is a third-class mixed pattern, not a second-class pure-0/2 pattern.
"""
function irregular_m3_patterns()
    bits = "111100000011001110"
    occupied = BitVector(c == '1' for c in bits)
    @assert count(occupied) == 9
    cell_n = [Int(occupied[2i-1])+Int(occupied[2i]) for i in 1:9]
    @assert (count(==(0),cell_n),count(==(1),cell_n),count(==(2),cell_n)) == (4,1,4)
    return [(; name="near02_irregular3", family="general", m=3,
               occupied, class_id=3)]
end
