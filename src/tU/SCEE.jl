function get_ABGM!(G1::G4Buffer_,G2::G4Buffer_,A::AreaBuffer_,B::AreaBuffer_,SCEE::SCEEBuffer_,nodes,idx,direction::String="Backward")
    G4!(SCEE,G1,nodes,idx,direction)
    G4!(SCEE,G2,nodes,idx,direction)
    GroverMatrix!(A.gmInv,view(G1.G0,A.index,A.index),view(G2.G0,A.index,A.index))
    A.detg=abs2(det(A.gmInv))
    LAPACK.getrf!(A.gmInv, A.ipiv)
    LAPACK.getri!(A.gmInv, A.ipiv)

    GroverMatrix!(B.gmInv,view(G1.G0,B.index,B.index),view(G2.G0,B.index,B.index))
    B.detg=abs2(det(B.gmInv))
    LAPACK.getrf!(B.gmInv, B.ipiv)
    LAPACK.getri!(B.gmInv, B.ipiv)
end 

function UpdateSCEELayer!(rng,s1,s2,G1::G4Buffer_,G2::G4Buffer_,A::AreaBuffer_,B::AreaBuffer_,model::tU_Hubbard_Para_,UPD::UpdateBuffer_,SCEE::SCEEBuffer_,λ)
    for i in axes(s1,1)
        UPD.subidx=[i]

        # update s1
        begin
            sx = rand(rng,  model.samplers_dict[s1[i]])
            p=get_r!(UPD,model.α * (model.η[sx]- model.η[s1[i]]),G1.Gt)
            p*=model.γ[sx]/model.γ[s1[i]]

            detTau_A=abs2(get_abTau1!(A,UPD,G2.G0,G1.Gt0,G1.G0t))
            detTau_B=abs2(get_abTau1!(B,UPD,G2.G0,G1.Gt0,G1.G0t))

            @fastmath p*= (detTau_A)^λ * (detTau_B)^(1-λ)
            if rand(rng)<p
                A.detg*=detTau_A
                B.detg*=detTau_B

                GMupdate!(A)
                GMupdate!(B)
                G4update!(SCEE,UPD,G1)
                s1[i]=sx
            end
        end

        # update ss[2]
        begin
            sx = rand(rng,  model.samplers_dict[s2[i]])
            p=get_r!(UPD,model.α * (model.η[sx]- model.η[s2[i]]),G2.Gt)
            p*=model.γ[sx]/model.γ[s2[i]]

            detTau_A=abs2(get_abTau2!(A,UPD,G1.G0,G2.Gt0,G2.G0t))
            detTau_B=abs2(get_abTau2!(B,UPD,G1.G0,G2.Gt0,G2.G0t))

            @fastmath p*= (detTau_A)^λ * (detTau_B)^(1-λ)
            if rand(rng)<p
                A.detg*=detTau_A
                B.detg*=detTau_B

                GMupdate!(A)
                GMupdate!(B)
                G4update!(SCEE,UPD,G2)
                s2[i]=sx
            end
        end
    end
end