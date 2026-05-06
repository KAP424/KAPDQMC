# Only PBC, OBC is not allowed

function Initial_Pt!(Lattice, Initial, Pt, K)
    Ns = size(K, 1)
    if Initial == "H0"
        KK = copy(K)
        μ = 1e-5
        KK .+= μ * diagm(repeat([-1, 1], div(Ns, 2)))
        E, V = LAPACK.syevd!('V', 'L', KK)
        Pt .= V[:, 1:div(Ns, 2)]
    elseif Initial == "V"
        if Lattice == "SQUARE90" || Lattice == "HoneyComb120" || Lattice == "HoneyComb60"
            for i in 1:div(Ns, 2)
                Pt[i*2, i] = 1
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
    end

end

"""
Only work for two-dimensional binary lattices
    convert (x,y) coordinate to even index
"""
function xy_i(Lattice::String, site::Vector{Int64}, x::Int64, y::Int64)::Int64
    if Lattice == "SQUARE90" || Lattice == "HoneyComb120" || Lattice == "HoneyComb60"
        if 0 > x > site[1] || 0 > y > site[2]
            error("Error : Out of Lattice Range!")
        end
        return 2 * (x + (y - 1) * site[1])
    elseif Lattice == "SQUARE45"
        if 0 > x > site[1] || 0 > y > site[2]
            error("Error : Out of Lattice Range!")
        end
        return x + (y - 1) * site[1]
    else
        error("Lattice: $(Lattice) is not allowed !")
    end
end
"""
Only work for two-dimensional binary lattices
    convert odd/even index to (x,y) coordinate
"""
function i_xy(Lattice::String, site::Vector{Int64}, i::Int64)
    if Lattice == "SQUARE90" || Lattice == "HoneyComb120" || Lattice == "HoneyComb60"
        j = Int(ceil(i / 2))
        return mod1(j, site[1]), Int(ceil(j / site[1]))
    elseif Lattice == "SQUARE45"
        return mod1(i, site[1]), Int(ceil(i / site[1]))
    else
        error("Lattice: $(Lattice) is not allowed !")
    end
end


function nnidx_F(Lattice, site)
    if Lattice == "SQUARE45"
        Ns = prod(site)
        if length(site) == 1
            nnidx = fill((0, 0), div(Ns, 2), 2)
            count = 1
            for i in 1:2:Ns
                nn = nn2idx(Lattice, site, i)
                for j in eachindex(nn)
                    nnidx[count, j] = (i, nn[j])
                end
                count += 1
            end
        elseif length(site) == 2
            nnidx = fill((0, 0), div(Ns, 2), 4)
            count = 1
            for x in 1:site[1]
                for y in 1:site[2]
                    if (x + y) % 2 == 1
                        i = x + (y - 1) * site[1]
                        nn = nn2idx(Lattice, site, i)
                        for j in eachindex(nn)
                            nnidx[count, j] = (i, nn[j])
                        end
                        count += 1
                    end
                end
            end
        end
    elseif Lattice == "SQUARE90" || occursin("HoneyComb", Lattice)
        Ns = prod(site) * 2
        nnidx = fill((0, 0), div(Ns, 2), length(nn2idx(Lattice, site, 1)))
        count = 1
        for i in 1:2:Ns
            nn = nn2idx(Lattice, site, i)
            for j in eachindex(nn)
                nnidx[count, j] = (i, nn[j])
            end
            count += 1
        end
    end
    return nnidx
end

"""
nearest neighbor indices
return 顺序:
    for HC: follow the same direction of A/B lattice
    for SQUARE: follow flux direction (different for A or B)
"""
function nn2idx(Lattice::String, site::Vector{Int64}, idx::Int64)
    if length(site) == 1
        nn = [mod1(idx - 1, site[1]), mod1(idx + 1, site[1])]
        return nn
    end
    x, y = i_xy(Lattice, site, idx)
    if Lattice == "SQUARE90"
        nn = zeros(Int, 4)
        if mod(idx, 2) == 1
            nn[1] = xy_i(Lattice, site, mod1(x - 1, site[1]), mod1(y - 1, site[2]))  #up
            nn[2] = idx + 1     # down
            nn[3] = xy_i(Lattice, site, mod1(x - 1, site[1]), y) #left
            nn[4] = xy_i(Lattice, site, x, mod1(y - 1, site[2])) #right
        else
            nn[1] = idx - 1     # up
            nn[2] = xy_i(Lattice, site, mod1(x + 1, site[1]), mod1(y + 1, site[2])) - 1  #down
            nn[3] = xy_i(Lattice, site, x, mod1(y + 1, site[2])) - 1 #left
            nn[4] = xy_i(Lattice, site, mod1(x + 1, site[1]), y) - 1 #right
        end
    elseif Lattice == "SQUARE45"
        nn = zeros(Int, 4)
        nn[1] = xy_i(Lattice, site, x, mod1(y + 1, site[2]))    #up
        nn[2] = xy_i(Lattice, site, x, mod1(y - 1, site[2]))    #down
        nn[3] = xy_i(Lattice, site, mod1(x - 1, site[1]), y)    #left
        nn[4] = xy_i(Lattice, site, mod1(x + 1, site[1]), y)    #right
    elseif Lattice == "HoneyComb120"
        nn = zeros(Int, 3)
        if mod(idx, 2) == 1
            nn[1] = idx + 1
            nn[2] = xy_i(Lattice, site, mod1(x + 1, site[1]), y)
            nn[3] = xy_i(Lattice, site, x, mod1(y - 1, site[2]))
        else
            nn[1] = idx - 1
            nn[2] = xy_i(Lattice, site, x, mod1(y + 1, site[2])) - 1
            nn[3] = xy_i(Lattice, site, mod1(x - 1, site[1]), y) - 1
        end

    elseif Lattice == "HoneyComb60"
        nn = zeros(Int, 3)
        if mod(idx, 2) == 1
            nn[1] = idx + 1
            nn[2] = xy_i(Lattice, site, mod1(x + 1, site[1]), mod1(y - 1, site[2]))
            nn[3] = xy_i(Lattice, site, x, mod1(y - 1, site[2]))

        else
            nn[1] = idx - 1
            nn[2] = xy_i(Lattice, site, x, mod1(y + 1, site[2])) - 1
            nn[3] = xy_i(Lattice, site, mod1(x - 1, site[1]), mod1(y + 1, site[2])) - 1
        end
    elseif Lattice == "triangular"
        nn = zeros(Int, 6)
        nn[1] = xy_i(Lattice, site, x, mod1(y + 1, site[2]))    #up
        nn[2] = xy_i(Lattice, site, x, mod1(y - 1, site[2]))    #down
        nn[3] = xy_i(Lattice, site, mod1(x - 1, site[1]), y)    #left
        nn[4] = xy_i(Lattice, site, mod1(x + 1, site[1]), y)    #right
        nn[5] = xy_i(Lattice, site, mod1(x - 1, site[1]), mod1(y + 1, site[2]))    #up-left
        nn[6] = xy_i(Lattice, site, mod1(x + 1, site[1]), mod1(y - 1, site[2]))    #down-right
    else
        error("Lattice: $(Lattice) is not allowed !")
    end
    return nn
end

"""
flux only work for SQUARE: 
    two options: "xy" and "y" for flux choices
anisotropy t only work for HoneyComb
"""
function nnK_Matrix(Lattice::String, site::Vector{Int64}; t=(1.0, 1.0, 1.0), flux=0.0, opt="xy")  # t for three directions
    flux1 = cis(flux / 4)
    flux2 = cis(-flux / 4)

    if Lattice == "SQUARE45"
        Ns = prod(site)
    else
        Ns = prod(site) * 2
    end

    if flux != 0.0 && opt != "y"
        K = zeros(ComplexF64, Ns, Ns)
    else
        K = zeros(Float64, Ns, Ns)
    end
    if occursin("SQUARE", Lattice)
        for i in 1:Ns
            nnidx = nn2idx(Lattice, site, i)
            if Lattice == "SQUARE90"
                if opt == "xy"
                    if mod(i, 2) == 1
                        K[i, nnidx[1]] = flux1
                        K[i, nnidx[2]] = flux1
                        K[i, nnidx[3]] = flux2
                        K[i, nnidx[4]] = flux2
                    else
                        K[i, nnidx[1]] = flux2
                        K[i, nnidx[2]] = flux2
                        K[i, nnidx[3]] = flux1
                        K[i, nnidx[4]] = flux1
                    end
                elseif opt == "y"
                    if mod(i, 2) == 1
                        K[i, nnidx[1]] = cis(flux)
                        K[i, nnidx[2]] = 1.0
                        K[i, nnidx[3]] = 1.0
                        K[i, nnidx[4]] = 1.0
                    else
                        K[i, nnidx[1]] = 1.0
                        K[i, nnidx[2]] = cis(flux)
                        K[i, nnidx[3]] = 1.0
                        K[i, nnidx[4]] = 1.0
                    end
                elseif opt == "yy"
                    if mod(i, 2) == 1
                        K[i, nnidx[1]] = cis(flux / 2)
                        K[i, nnidx[2]] = cis(flux / 2)
                        K[i, nnidx[3]] = 1.0
                        K[i, nnidx[4]] = 1.0
                    else
                        K[i, nnidx[1]] = cis(-flux / 2)
                        K[i, nnidx[2]] = cis(-flux / 2)
                        K[i, nnidx[3]] = 1.0
                        K[i, nnidx[4]] = 1.0
                    end
                end
            elseif Lattice == "SQUARE45"
                x, y = i_xy(Lattice, site, i)
                if mod(x + y, 2) == 1
                    K[i, nnidx[1]] = flux1
                    K[i, nnidx[2]] = flux1
                    K[i, nnidx[3]] = flux2
                    K[i, nnidx[4]] = flux2
                else
                    K[i, nnidx[1]] = flux2
                    K[i, nnidx[2]] = flux2
                    K[i, nnidx[3]] = flux1
                    K[i, nnidx[4]] = flux1
                end
            end
        end
    elseif occursin("HoneyComb", Lattice)
        if flux != 0.0
            error("Warning: flux is not implemented for HoneyComb lattice!")
        end
        for i in 1:Ns
            nnidx = nn2idx(Lattice, site, i)
            for j in eachindex(nnidx)
                K[i, nnidx[j]] = t[j]
            end
        end
    end
    @assert norm(K - K') < 1e-8 "K is not Hermitian!"
    return K
end

function area_index(Lattice::String, site::Vector{Int64}, area::Tuple{Vector{Int64},Vector{Int64}})::Vector{Int64}
    if length(site) == 1
        index = [x for x in area[1][1]:area[2][1]]
        return index
    end
    if Lattice == "SQUARE45"
        counter = 1
        index = zeros(Int64, prod(area[2] - area[1] + [1, 1]))
        for ly in area[1][2]:area[2][2]
            for lx in area[1][1]:area[2][1]
                index[counter] = xy_i(Lattice, site, lx, ly)
                counter += 1
            end
        end
        return index
    elseif Lattice == "SQUARE90"
        counter = 1
        index = zeros(Int64, 2 * prod(area[2] - area[1] + [1, 1]))
        for lx in area[1][1]:area[2][1]
            for ly in area[1][2]:area[2][2]
                index[counter] = xy_i(Lattice, site, lx, ly) - 1
                index[counter+1] = index[counter] + 1
                counter += 2
            end
        end
        return index
    elseif occursin("HoneyComb", Lattice)
        L = site[1]
        if area[1][1] == -1
            if Lattice == "HoneyComb60"
                println("zigzag")
                index = collect(4:2:xy_i(Lattice, site, L - 1, 1))
                for i in 2:div(2 * L, 3)
                    index = vcat(collect(xy_i(Lattice, site, 2, i)-1:1:xy_i(Lattice, site, L - i, i)), index)
                end
                return index
            else
                error("zigzag Only for HoneyComb60°")
            end

        elseif area[1][1] == -2
            if Lattice == "HoneyComb60"
                index = Vector{Int64}()
                println("beared")
                for i in 2:div(2 * L, 3)
                    index = vcat(xy_i(Lattice, site, 2, i) - 1, index)
                    index = vcat(collect(xy_i(Lattice, site, 3, i)-1:1:xy_i(Lattice, site, L - i + 1, i)-1), index)
                end
                index = vcat(xy_i(Lattice, site, 2, div(2 * L, 3) + 1) - 1, index)
                return index
            else
                error("beared Only for HoneyComb60°")
            end
        else
            counter = 1
            index = zeros(Int64, 2 * prod(area[2] - area[1] + [1, 1]))
            for lx in area[1][1]:area[2][1]
                for ly in area[1][2]:area[2][2]
                    index[counter] = xy_i(Lattice, site, lx, ly) - 1
                    index[counter+1] = index[counter] + 1
                    counter += 2
                end
            end
            return index
        end
    else
        error("Lattice: $(Lattice) is not allowed !")
    end
end

function name_Lattice(Lattice::String)
    if Lattice == "SQUARE90"
        return "□90"
    elseif Lattice == "SQUARE45"
        return "□45"
    elseif Lattice == "HoneyComb60"
        return "HC60"
    elseif Lattice == "HoneyComb120"
        return "HC120"
    else
        error("Lattice: $(Lattice) is not allowed !")
    end
end

"""
next nearest neighbor indices
"""
function nnn2idx(Lattice::String, site::Vector{Int64}, idx::Int64)
    if Lattice == "HoneyComb120"
        nnn = zeros(Int, 6)
        x, y = i_xy(Lattice, site, idx)
        if mod(idx, 2) == 1
            nnn[1] = xy_i(Lattice, site, x, mod1(y + 1, site[2])) - 1
            nnn[2] = xy_i(Lattice, site, mod1(x + 1, site[1]), y) - 1
            nnn[3] = xy_i(Lattice, site, mod1(x + 1, site[1]), mod1(y + 1, site[2])) - 1
            nnn[4] = xy_i(Lattice, site, mod1(x - 1, site[1]), y) - 1
            nnn[5] = xy_i(Lattice, site, mod1(x - 1, site[1]), mod1(y - 1, site[2])) - 1
            nnn[6] = xy_i(Lattice, site, x, mod1(y - 1, site[2])) - 1
        else
            nnn[1] = xy_i(Lattice, site, x, mod1(y + 1, site[2]))
            nnn[2] = xy_i(Lattice, site, mod1(x + 1, site[1]), y)
            nnn[3] = xy_i(Lattice, site, mod1(x + 1, site[1]), mod1(y + 1, site[2]))
            nnn[4] = xy_i(Lattice, site, mod1(x - 1, site[1]), y)
            nnn[5] = xy_i(Lattice, site, mod1(x - 1, site[1]), mod1(y - 1, site[2]))
            nnn[6] = xy_i(Lattice, site, x, mod1(y - 1, site[2]))
        end
    elseif Lattice == "HoneyComb60"
        nnn = zeros(Int, 3)
        x, y = i_xy(Lattice, site, idx)
        if mod(idx, 2) == 1
            nnn[1] = xy_i(Lattice, site, x, mod1(y + 1, site[2])) - 1
            nnn[2] = xy_i(Lattice, site, mod1(x + 1, site[1]), y) - 1
            nnn[3] = xy_i(Lattice, site, mod1(x + 1, site[1]), mod1(y - 1, site[2])) - 1
            nnn[4] = xy_i(Lattice, site, mod1(x - 1, site[1]), y) - 1
            nnn[5] = xy_i(Lattice, site, mod1(x - 1, site[1]), mod1(y + 1, site[2])) - 1
            nnn[6] = xy_i(Lattice, site, x, mod1(y - 1, site[2])) - 1
        else
            nnn[1] = xy_i(Lattice, site, x, mod1(y + 1, site[2]))
            nnn[2] = xy_i(Lattice, site, mod1(x + 1, site[1]), y)
            nnn[3] = xy_i(Lattice, site, mod1(x + 1, site[1]), mod1(y - 1, site[2]))
            nnn[4] = xy_i(Lattice, site, mod1(x - 1, site[1]), y)
            nnn[5] = xy_i(Lattice, site, mod1(x - 1, site[1]), mod1(y + 1, site[2]))
            nnn[6] = xy_i(Lattice, site, x, mod1(y - 1, site[2]))
        end
    else
        error("Lattice: $(Lattice) is not allowed !")
    end
    return nnn
end


function nnnK_Matrix(Lattice::String, site::Vector{Int64})
    if Lattice == "SQUARE"
        Ns = prod(site)
        K = zeros(Float64, Ns, Ns)

    elseif occursin("HoneyComb", Lattice)
        Ns = prod(site) * 2
        K = zeros(Float64, Ns, Ns)
    end

    for i in 1:Ns
        nnnidx = nnn2idx(Lattice, site, i)
        for idx in nnnidx
            K[i, idx] = 1
        end
    end
    return K
end

"""
the third nearest neighbor indices
"""
function n3n2idx(Lattice::String, site::Vector{Int64}, idx::Int64)
    if Lattice == "HoneyComb120"
        n3n = zeros(Int, 3)
        x, y = i_xy(Lattice, site, idx)
        if mod(idx, 2) == 1
            n3n[1] = xy_i(Lattice, site, mod1(x + 1, site[1]), mod1(y + 1, site[2]))
            n3n[2] = xy_i(Lattice, site, mod1(x + 1, site[1]), mod1(y - 1, site[2]))
            n3n[3] = xy_i(Lattice, site, mod1(x - 1, site[1]), mod1(y - 1, site[2]))
        else
            n3n[1] = xy_i(Lattice, site, mod1(x + 1, site[1]), mod1(y + 1, site[2])) - 1
            n3n[2] = xy_i(Lattice, site, mod1(x - 1, site[1]), mod1(y - 1, site[2])) - 1
            n3n[3] = xy_i(Lattice, site, mod1(x - 1, site[1]), mod1(y + 1, site[2])) - 1
        end
    elseif Lattice == "HoneyComb60"
        n3n = zeros(Int, 3)
        x, y = i_xy(Lattice, site, idx)
        if mod(idx, 2) == 1
            n3n[1] = xy_i(Lattice, site, mod1(x + 1, site[1]), y)
            n3n[2] = xy_i(Lattice, site, mod1(x - 1, site[1]), y)
            n3n[3] = xy_i(Lattice, site, mod1(x + 1, site[1]), mod1(y - 2, site[2]))
        else
            n3n[1] = xy_i(Lattice, site, mod1(x - 1, site[1]), y) - 1
            n3n[2] = xy_i(Lattice, site, mod1(x + 1, site[1]), y) - 1
            n3n[3] = xy_i(Lattice, site, mod1(x - 1, site[1]), mod1(y + 2, site[2])) - 1
        end
    else
        error("Lattice: $(Lattice) is not allowed !")
    end
    return n3n
end

function n3nK_Matrix(Lattice::String, site::Vector{Int64})
    if occursin("HoneyComb", Lattice)
        Ns = prod(site) * 2
        K = zeros(Float64, Ns, Ns)
    else
        error("Lattice: $(Lattice) is not allowed !")
    end

    for i in 1:Ns
        n3nidx = n3n2idx(Lattice, site, i)
        for idx in n3nidx
            K[i, idx] = 1
        end
    end
    return K
end


function E_piflux(kx, ky, flux)
    return 2 * (sqrt(cos(kx)^2 + cos(ky)^2 + 2 * cos(kx) * cos(ky) * cos(2 * flux / 4)))
end

function E_piflux2(kx, ky)
    return 2 * (sqrt(cos(kx / 2)^2 + cos(ky)^2))
end

if PROGRAM_FILE == @__FILE__
    using LinearAlgebra
    Lattice = "SQUARE45"
    site = [6, 6]
    idx = area_index(Lattice, site, ([1, 1], [6, 3]))
    K = nnK_Matrix(Lattice, site, flux=π)
    # println(nn2idx(Lattice, site, 1))
    println((idx))

    Lattice = "SQUARE90"
    site = [4, 4]
    K = nnK_Matrix(Lattice, site, flux=π)
    # idx = area_index(Lattice, site, ([1, 1], [4, 2]))
    # println(nn2idx(Lattice, site, 11))
    # println((idx))

    # K1, K2 = nnK_Matrix(Lattice, site, flux=π)
    # E1, V1 = LAPACK.syevd!('V', 'L', K1[:, :])
    # E2, V2 = LAPACK.syevd!('V', 'L', K2[:, :])
    # println(norm(E1 - E2))
    # println(norm(V1 - V2))
    # println(size(K))
    # println(length(E))

    # kx = (collect(1:1:site[1]) / site[1] .- 1 / 2) * 2π
    # ky = (collect(1:1:site[2]) / site[2] .- 1 / 2) * 2π
    # EE = zeros(Float64, length(kx), length(ky))
    # for i in eachindex(kx)
    #     for j in eachindex(ky)
    #         EE[i, j] = E_piflux(kx[i], ky[j], π)
    #         # EE[i, j] = E_piflux2(kx[i], ky[j])
    #     end
    # end
    # EE = sort(vec(EE))

    # # println(E[div(length(E), 2)+1:end])
    # # println(EE)

    # println(norm(E[div(length(E), 2)+1:end] - EE))
end
