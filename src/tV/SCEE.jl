function get_ABGM!(G1::G4Buffer_,G2::G4Buffer_,A::AreaBuffer_,B::AreaBuffer_,SCEE::SCEEBuffer_,nodes,idx,direction::String="Backward")
    G4!(SCEE,G1,nodes,idx,direction)
    G4!(SCEE,G2,nodes,idx,direction)
    GroverMatrix!(A.gmInv,view(G1.G0,A.index,A.index),view(G2.G0,A.index,A.index))
    A.detg=abs(det(A.gmInv))
    LAPACK.getrf!(A.gmInv, A.ipiv)
    LAPACK.getri!(A.gmInv, A.ipiv)

    GroverMatrix!(B.gmInv,view(G1.G0,B.index,B.index),view(G2.G0,B.index,B.index))
    B.detg=abs(det(B.gmInv))
    LAPACK.getrf!(B.gmInv, B.ipiv)
    LAPACK.getri!(B.gmInv, B.ipiv)
end 