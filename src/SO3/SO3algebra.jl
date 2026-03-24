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
            # count = 1
            # for i in 1:Ns
            #     x, y = i_xy(Lattice, site, i)
            #     if (x + y) % 2 == 1
            #         Pt[i, count] = 1
            #         count += 1
            #         if count > div(Ns, 2)
            #             break
            #         end
            #     end
            # end
            error("Initial state $Initial not supported for Lattice $Lattice!")
        end
    elseif Initial == "HJ"
        HJ = zeros(ComplexF64, size(K))
        hJ = [0 1im 0; -1im 0 0; 0 0 0]
        for i in 1:div(Ns, 3)
            if i % 2 == 1
                HJ[3*(i-1)+1:3*i, 3*(i-1)+1:3*i] .= hJ
            else
                HJ[3*(i-1)+1:3*i, 3*(i-1)+1:3*i] .= -hJ
            end
        end
        HJ .+= 1e-5 * diagm(repeat([-1, -1, -1, 1, 1, 1], div(Ns, 6)))
        E, V = LAPACK.syevd!('V', 'L', HJ)
        Pt .= V[:, 1:div(Ns, 2)]
    else
        error("Initial state $Initial not supported!")
    end
    @assert norm(Pt' * Pt - I(div(Ns, 2))) < 1e-10 "Pt is not unitary!"
end


function so3bondTidx_F(Lattice, site)
    function so3bondTidx(i, bond)
        if bond == 1
            return (i - 1) * 3 + 2, (i - 1) * 3 + 3
        elseif bond == 2
            return (i - 1) * 3 + 1, (i - 1) * 3 + 3
        elseif bond == 3
            return (i - 1) * 3 + 1, (i - 1) * 3 + 2
        end
    end
    if occursin("HoneyComb", Lattice)
        Ns = 2 * prod(site)
    end
    nnidx = fill((0, 0), Ns, 3)

    for i in 1:Ns
        for bond in 1:3
            nnidx[i, bond] = so3bondTidx(i, bond)
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

function xyzσTidx(Lattice, site, x, y, z, σ)
    """
    [x,y]: site coordinate
    z: A/B sublattice index (1,2)
    σ: flavor type (1,2,3)
    """
    idx = xy_i(Lattice, site, x, y)
    if z == 1
        idx -= 1
    end
    idx = 3 * idx - 3 + σ
    return idx
end

function so3area_index(Lattice::String, site::Vector{Int64}, area::Tuple{Vector{Int64},Vector{Int64}})::Vector{Int64}
    index = area_index(Lattice, site, area)
    so3index = zeros(Int64, 3 * length(index))
    for i in eachindex(index)
        for σ in 1:3
            so3index[3*(i-1)+σ] = 3 * index[i] - 3 + σ
        end
    end
    return so3index
end

