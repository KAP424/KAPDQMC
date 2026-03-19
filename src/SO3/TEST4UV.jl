if abspath(PROGRAM_FILE) == @__FILE__

    push!(LOAD_PATH, "D:\\JuliaDQMC\\code\\KAPDQMC\\src\\")
    using KAPDQMC
    using LinearAlgebra, Random

    include("GreenMatrix.jl")
    include("model.jl")


    Lattice = "HoneyComb120"
    site = [3, 3]
    K = nnK_Matrix4so3(Lattice, site)

    so3idx = so3Tindex_F(Lattice, site)

    model = SO3_Hubbard_Para(Ht=1.0, HJ1=2.0, HJ2=2.0,
        Θrelax=3.0, Θquench=0.0, Lattice=Lattice, site=site, Δt=0.05, BatchSize=5, Initial="H0")

    rng = MersenneTwister(time_ns())

    s = SO3Initial_s(model, rng)
    # println("size(s): ", size(s))
    # println("size(model.bondidx): ", size(model.bondidx))

    @assert model.K[]

    # ------------------------------------------------------------------------------------------------------------
    tmpVV = zeros(ComplexF64, size(model.bondidx, 2), model.Ns, model.Ns)
    for lt in 1:model.Nt
        for bond in 1:size(model.bondidx, 2)
            tmpN = zeros(ComplexF64, model.Ns)
            for i in 1:size(model.bondidx, 1)
                x, y = model.bondidx[i, bond]
                tmpVV[bond, x, y] = model.α[lt] * model.η[s[i, bond, lt]]
                tmpVV[bond, y, x] = -model.α[lt] * model.η[s[i, bond, lt]]
                tmpN[x] = -1im * model.α[lt] * model.η[s[i, bond, lt]]
                tmpN[y] = 1im * model.α[lt] * model.η[s[i, bond, lt]]
            end
            @assert (norm(model.UV[:, :, bond]' * model.UV[:, :, bond] - I(model.Ns))) < 1e-5
            @assert (norm(model.UV[:, :, bond] * model.UV[:, :, bond]' - I(model.Ns))) < 1e-5
            @assert (norm(model.UV[:, :, bond] * Diagonal(tmpN) * model.UV[:, :, bond]' - tmpVV[bond, :, :])) < 1e-5
        end
    end

    # ------------------------------------------------------------------------------------------------------------
    # println(model.α)
    @assert norm(model.eK * model.eKinv - I(model.Ns)) < 1e-10 "eK*eKinv does not equal identity!"

    BM = zeros(ComplexF64, model.Ns, model.Ns)
    BMinv = zeros(ComplexF64, model.Ns, model.Ns)
    tmpN = zeros(ComplexF64, model.Ns)
    tmpNN = zeros(ComplexF64, model.Ns, model.Ns)

    for i in 1:length(model.nodes)-1
        BM_F!(tmpN, tmpNN, BM, model, s, i)
        BMinv_F!(tmpN, tmpNN, BMinv, model, s, i)
        @assert (norm(BM * BMinv - I(model.Ns))) < 1e-10 "BM*BMinv does not equal identity!"
    end
    # ------------------------------------------------------------------------------------------------------------
    UPD = UpdateBuffer()
    s1 = SO3Initial_s(model, rng)
    # flip a single bond
    lt = 1
    site_idx = 3

    for bond in 1:size(model.bondidx, 2)
        s2 = copy(s1)
        s2[site_idx, bond, lt] = rand(model.samplers_dict[s1[site_idx, bond, lt]])

        tmpV1 = zeros(ComplexF64, model.Ns, model.Ns)
        tmpV2 = zeros(ComplexF64, model.Ns, model.Ns)
        for i in 1:size(model.bondidx, 1)
            x, y = model.bondidx[i, bond]
            tmpV1[x, y] = model.α[lt] * model.η[s1[i, bond, lt]]
            tmpV1[y, x] = -model.α[lt] * model.η[s1[i, bond, lt]]
            tmpV2[x, y] = model.α[lt] * model.η[s2[i, bond, lt]]
            tmpV2[y, x] = -model.α[lt] * model.η[s2[i, bond, lt]]
        end
        x, y = model.bondidx[site_idx, bond]
        diffV = (tmpV1-tmpV2)[[x, y], [x, y]]
        println("Difference in tmpV for flipped bond: ", diffV)

        Δs = model.η[s2[site_idx, bond, lt]] - model.η[s1[site_idx, bond, lt]]

        # println("Difference in s: ", Δs)
        # println("Difference in s: ", model.α[lt] * Δs)

        UPD.tmp2 .= model.α[lt] * Δs .* [1im, -1im]


        @assert norm(UPD.uv * Diagonal(UPD.tmp2) * UPD.uv' - diffV) < 1e-5 "Transformed difference does not match expected diffV!"
        # println("Transformed difference: ", diag(UPD.uv' * diffV * UPD.uv))

    end


end


