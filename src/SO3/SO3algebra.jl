
function SO3Initial_Pt!(Lattice, Initial, Pt, K)
    Ns = size(K, 1)
    if Initial == "H0"
        KK = copy(K)
        μ = 1e-5
        KK .+= μ * diagm(repeat([-1, -1, -1, 1, 1, 1], div(Ns, 6)))
        E, V = LAPACK.syevd!('V', 'L', KK)
        Pt .= V[:, 1:div(Ns, 2)]
    elseif Initial == "V"
        if Lattice == "SQUARE90" || Lattice == "HoneyComb120" || Lattice == "HoneyComb60"
            count = 1
            for i in 1:div(Ns, 6)
                Pt[(i-1)*6+1, count] = 1
                Pt[(i-1)*6+2, count+1] = 1
                Pt[(i-1)*6+3, count+2] = 1
                count += 3
            end
        elseif Lattice == "SQUARE45"
            count = 1
            for i in 1:Ns
                x, y = i_xy(Lattice, site, i)
                if (x + y) % 2 == 1
                    Pt[i, count] = 1
                    count += 1
                    if count > div(Ns, 2)
                        break
                    end
                end
            end
        end
    elseif Initial == "HJ"

    else
        error("Initial state $Initial not supported!")
    end

end

function so3Tindex(site, bond)
    if bond == 1
        return (site - 1) * 3 + 2, (site - 1) * 3 + 3
    elseif bond == 2
        return (site - 1) * 3 + 1, (site - 1) * 3 + 3
    elseif bond == 3
        return (site - 1) * 3 + 1, (site - 1) * 3 + 2
    end
end

function so3Tindex_F(Lattice, site)
    if occursin("HoneyComb", Lattice)
        Ns = 2 * prod(site)
    end
    nnidx = fill((0, 0), Ns, 3)

    for i in 1:Ns
        for bond in 1:3
            nnidx[i, bond] = so3Tindex(i, bond)
        end
    end
    return nnidx
end

function nnK_Matrix4so3(Lattice, site, flux=0.0, opt="xy")
    if Lattice == "SQUARE90"
        error("Lattice $Lattice not supported!")
    elseif Lattice == "HoneyComb120"
        kk = nnK_Matrix(Lattice, site, flux=flux, opt=opt)

        N = size(kk, 1)
        K = zeros(ComplexF64, 3 * N, 3 * N)

        K[1:3:end, 1:3:end] = kk
        K[2:3:end, 2:3:end] = kk
        K[3:3:end, 3:3:end] = kk

    else
        error("Lattice $Lattice not supported!")
    end
    return K
end

