#!/usr/bin/env julia
#=
Strong-measurement entanglement scaling (lean pipeline).

Fixed finite L, fixed initial pattern P, pre-measurement state |psi(theta=inf)>.
The post-measurement Renyi-2 entropy has the exact fixed-L expansion

    S2(R; beta) = A_R(P,R) e^{-4 beta} + O(e^{-8 beta}),

so A_R is a beta independent functional of the pre-measurement state.  We compute
A_R exactly for the three scaling regions

    A = (L/3) x L,   C = L x (L/3),   B = (L/3) x (2L/3),
    D_A = (2 A_A + A_C)/3 - A_B,

and ask how A_R and D_A scale with L.  D_A is the same corner subtraction as the
equilibrium diagnostic D, applied to the susceptibility A_R instead of to the
entropy itself.

Two independent routes to A_R:

  dense  : full real-space orbital matrix (2L^2 x L^2) plus a dense solve
           X = Q_E / Q_F.  Memory O(L^4), time O(L^6).  Oracle only.
  kernel : X(k) = q_k[E] / q_k[F] in each supercell momentum block (m^2 x m^2),
           then a separable Fourier transform to X(delta).  No L^2 x L^2 object
           is ever formed; region sums become indicator cross-correlations.
           Memory O(m^4 q^2), time O(m^4 q^3).

The kernel route is the production path.  It reaches L = 216-288 in seconds,
which is what makes the L-scaling question answerable at all.

Knobs (environment): SS_PATH, SS_PATTERNS, SS_LADDER, SS_XCHECK_LMAX, SS_BETA,
SS_DENSE_ORACLE_LMAX.
=#
using LinearAlgebra, Printf

const HERE = @__DIR__
include(joinpath(HERE, "fermion_experiment.jl"))
const FE = FermionExperiment
include(joinpath(HERE, "experiment_patterns.jl"))

# ---------------------------------------------------------------- DFT helpers
const _DFT = Dict{Int,Matrix{ComplexF64}}()
"""Unnormalised forward DFT matrix: Ahat[k] = sum_r A[r] exp(-2 pi i k r / q)."""
dftmat(q::Int) = get!(_DFT, q) do
    [cis(-2pi * r * k / q) for r in 0:q-1, k in 0:q-1]
end
dft2(a) = (F = dftmat(size(a, 1)); F * a * transpose(F))

"""
    corr_sum(gh, uh, vh, q)

`sum_delta G(delta) sum_r U(r) V(r - delta)` from precomputed DFTs, by Parseval.
"""
function corr_sum(gh, uh, vh, q)
    s = 0.0
    @inbounds for j in 1:q, i in 1:q
        s += real(conj(gh[i, j]) * uh[i, j] * conj(vh[i, j]))
    end
    return s / q^2
end

"""
    pair_sum(G, U, V)

Same sum for untransformed real q x q arrays.  `G = |X_ab(delta)|^2`, `U`/`V`
are region indicators.  The L^2 x L^2 object is never built.
"""
pair_sum(G, U, V) = corr_sum(dft2(G), dft2(U), dft2(V), size(G, 1))

"""Cheap internal guard: pair_sum must equal the naive double sum."""
function selftest()
    q = 6
    G = [cos(2pi * (i * 0.37 + j * 0.91) / q) for i in 0:q-1, j in 0:q-1]
    U = BitMatrix([(i + j) % 3 == 0 for i in 0:q-1, j in 0:q-1])
    V = BitMatrix([(2i + j) % 4 == 1 for i in 0:q-1, j in 0:q-1])
    naive = 0.0
    for i in 0:q-1, j in 0:q-1, r in 0:q-1, s in 0:q-1
        (U[r+1, s+1] && V[mod(r-i, q)+1, mod(s-j, q)+1]) || continue
        naive += G[i+1, j+1]
    end
    return abs(pair_sum(G, U, V) - naive) / max(abs(naive), 1.0)
end

# ------------------------------------------------------- real-space X kernel
"""
    x_kernel(state) -> (kernel, eset, fset, min_svd)

`kernel[a,b,dx+1,dy+1] = X_{a,b}(delta)` with `X = Q_E Q_F^{-1}`, indexed by
supercell displacement and by within-supercell site type.  The pattern is
supercell periodic, so `X` is block diagonal in supercell momentum and every
block is an `m^2 x m^2` solve.
"""
function x_kernel(state)
    model = state.prepared.model
    m, q = model.m, model.q
    occ = state.prepared.occupied
    eset, fset = findall(.!occ), findall(occ)
    blocks = Vector{Matrix{ComplexF64}}(undef, q * q)
    minsv = Inf
    for k in 1:q*q
        u = state.orbitals[k]
        qf = u[fset, :]
        minsv = min(minsv, minimum(svdvals(qf)))
        blocks[k] = u[eset, :] / qf
    end
    Finv = conj.(dftmat(q)) ./ q          # x(delta) = (1/q^2) sum_k X(k) e^{+i k.delta}
    spec = Matrix{ComplexF64}(undef, q, q)
    kernel = Array{ComplexF64}(undef, length(eset), length(fset), q, q)
    for b in eachindex(fset), a in eachindex(eset)
        for ky in 0:q-1, kx in 0:q-1
            spec[kx+1, ky+1] = blocks[1+kx+q*ky][a, b]
        end
        kernel[a, b, :, :] = Finv * spec * transpose(Finv)
    end
    return kernel, eset, fset, minsv
end

"""Region membership as one q x q indicator per within-supercell site type."""
function region_masks(model, width::Int, height::Int; x0=0, y0=0)
    m, q = model.m, model.q
    masks = [falses(q, q) for _ in 1:2m^2]
    for Y in 0:q-1, X in 0:q-1, iy in 0:m-1, ix in 0:m-1
        x, y = m * X + ix, m * Y + iy
        (x0 <= x <= x0 + width - 1 && y0 <= y <= y0 + height - 1) || continue
        for s in 1:2
            masks[s+2*(ix+m*iy)][X+1, Y+1] = true
        end
    end
    return masks
end

"""`A_R = 2(||X[E in R, F outside R]||^2 + ||X[E outside R, F in R]||^2)`."""
function aR_kernel(kernel, eset, fset, masks)
    q = size(kernel, 3)
    inside = [dft2(m) for m in masks]
    outside = [dft2(.!m) for m in masks]
    total = 0.0
    for (ai, a) in enumerate(eset), (bi, b) in enumerate(fset)
        gh = dft2(abs2.(kernel[ai, bi, :, :]))
        total += corr_sum(gh, inside[a], outside[b], q)
        total += corr_sum(gh, outside[a], inside[b], q)
    end
    return 2total
end

# ------------------------------------------------------- dense reference route
function full_orbitals(state)
    model = state.prepared.model
    p = model.m^2
    out = zeros(ComplexF64, 2model.L^2, model.L^2)
    for i in axes(out, 1)
        a, rx, ry = FE.site_coordinates(model, i)
        for ky in 0:model.q-1, kx in 0:model.q-1
            k = 1 + kx + model.q * ky
            phase = cis(2pi * (rx * kx + ry * ky) / model.q) / model.q
            out[i, (k-1)*p+1:k*p] = phase .* state.orbitals[k][a, :]
        end
    end
    return out
end

"""Dense oracle: `X = Q_E / Q_F` from the full 2L^2 x L^2 orbital matrix."""
function dense_x(state)
    model = state.prepared.model
    q = full_orbitals(state)
    f = FE.occupied_indices(state.prepared)
    e = setdiff(1:2model.L^2, f)
    return (x=q[e, :] / q[f, :], f=f, e=e, smin=minimum(svdvals(q[f, :])))
end

function aR_dense(dx, indices)
    region = Set(indices)
    fr = map(in(region), dx.f)
    er = map(in(region), dx.e)
    return 2(sum(abs2, dx.x[er, .!fr]) + sum(abs2, dx.x[.!er, fr]))
end

# ------------------------------------------------------------------- regions
region_specs(L::Int) = (
    A=(width=L ÷ 3, height=L),
    C=(width=L, height=L ÷ 3),
    B=(width=L ÷ 3, height=2L ÷ 3),
)

# ------------------------------------------------------------------ fittings
"""Least squares y = a x + b; returns (a, b, rms)."""
function lsq(x, y)
    A = [x ones(length(x))]
    c = A \ y
    r = y - A * c
    return c[1], c[2], sqrt(sum(abs2, r) / length(x))
end

"""
    corner_fit(L, y; lmin)

Logarithmic corner coefficient with the same three-parameter model used by the
frozen finite-beta analysis: `y = c ln L + b + d/L`, restricted to `L >= lmin`.
"""
function corner_fit(L, y; lmin=36.0)
    keep = L .>= lmin
    count(keep) >= 3 || return (NaN, NaN, NaN, NaN)
    l, yy = L[keep], y[keep]
    A = [log.(l) ones(length(l)) 1 ./ l]
    c = A \ yy
    r = yy - A * c
    return (c[1], c[2], c[3], sqrt(sum(abs2, r) / length(l)))
end

# Frozen equilibrium finite-window log coefficients at fixed beta, from
# RESULTS.md ("63 组主拟合"; three-parameter log+const+1/L fit, L = 36-216).
# External target only: nothing below is used to build A_R.
const FROZEN_C = Dict(
    "one_B_2" => [(1.0, 0.06742882), (1.25, 0.03642392), (1.5, 0.01627426),
        (1.75, 0.006505095), (2.0, 0.002472039)],
    "diagonal_B_3" => [(1.0, 0.08403503), (1.25, 0.04639887), (1.5, 0.02070907),
        (1.75, 0.008218309), (2.0, 0.003108257)],
    "general_matched_a" => [(1.0, 0.0535626), (1.25, 0.02355965), (1.5, 0.009331646),
        (1.75, 0.003532237), (2.0, 0.001313401)],
    "pure02_block4" => [(1.0, 0.1037911), (1.25, 0.05981079), (1.5, 0.02801958),
        (1.75, 0.01149776), (2.0, 0.004420536)],
)

# ---------------------------------------------------------------- main driver
function main()
    path = get(ENV, "SS_PATH", "dense")
    pattern_names = split(get(ENV, "SS_PATTERNS", "uniform_A,one_B_2"), ',')
    ladder = parse.(Int, split(get(ENV, "SS_LADDER", "12,18,24,30,36,42,48"), ','))
    xcheck_lmax = parse(Int, get(ENV, "SS_XCHECK_LMAX", "48"))
    beta_x = parse(Float64, get(ENV, "SS_BETA", "4.0"))
    oracle_lmax = parse(Int, get(ENV, "SS_DENSE_ORACLE_LMAX", "0"))
    catalogue = Dict(p.name => p for p in experiment_patterns())

    st = selftest()
    st < 1e-12 || error("pair_sum self-test failed: $st")
    println("pair_sum self-test rel. error: $(st)")

    rows = NamedTuple[]
    oracle_err = 0.0
    for name in pattern_names
        p = catalogue[name]
        for L in ladder
            # Protocol commensurability (neq.md): L must be a common multiple of
            # the pattern cell m and of 3, i.e. L = 3m * integer, so that the
            # region edges fall on supercell boundaries.  Incommensurate sizes
            # give correct A_R but for a region that cuts through supercells, and
            # their values are not comparable across L (observed in run 2).
            L % (3p.m) == 0 || continue
            model = FE.BlochModel(L, p.m)
            prep = FE.prepare_pattern(model, p.occupied)
            base = FE.evolve(prep, Inf)
            specs = region_specs(L)
            vals = Dict{Symbol,Float64}()
            if path == "dense"
                dx = dense_x(base)
                smin = dx.smin
                for (key, sp) in pairs(specs)
                    vals[key] = aR_dense(dx, FE.rectangle_indices(L; width=sp.width, height=sp.height))
                end
            else
                kernel, eset, fset, smin = x_kernel(base)
                for (key, sp) in pairs(specs)
                    vals[key] = aR_kernel(kernel, eset, fset, region_masks(model, sp.width, sp.height))
                end
                if L <= oracle_lmax
                    dx = dense_x(base)
                    for (key, sp) in pairs(specs)
                        ref = aR_dense(dx, FE.rectangle_indices(L; width=sp.width, height=sp.height))
                        oracle_err = max(oracle_err, abs(vals[key] / ref - 1))
                    end
                end
            end
            xcheck = NaN
            if L <= xcheck_lmax
                meas = FE.measure(base, beta_x)
                s = Dict(:A => FE.entropy_strip(meas; width=L ÷ 3),
                    :C => FE.entropy_strip(meas; width=L ÷ 3, orientation=:y),
                    :B => FE.entropy_rectangle(meas))
                xcheck = maximum(abs(s[k] * exp(4beta_x) / vals[k] - 1) for k in (:A, :C, :B))
            end
            push!(rows, (; pattern=name, m=p.m, L=L, path=path, smin=smin,
                A_A=vals[:A], A_C=vals[:C], A_B=vals[:B],
                D_A=(2vals[:A] + vals[:C]) / 3 - vals[:B], xcheck=xcheck))
            @printf("  %-18s L=%-4d A_A=%12.6f A_B=%12.6f D_A=%10.6f%s\n",
                name, L, vals[:A], vals[:B], (2vals[:A] + vals[:C]) / 3 - vals[:B],
                isnan(xcheck) ? "" : @sprintf("  xcheck=%.2e", xcheck))
            flush(stdout)
        end
    end

    # ---- frozen small-L reference (data/strong_measurement_controls.csv)
    ref_err = 0.0
    ref_n = 0
    refcsv = joinpath(HERE, "data", "strong_measurement_controls.csv")
    if isfile(refcsv)
        lines = readlines(refcsv)
        pos = Dict(h => i for (i, h) in enumerate(split(lines[1], ',')))
        for line in lines[2:end]
            f = split(line, ',')
            nm, L = String(f[pos["pattern"]]), parse(Int, f[pos["L"]])
            ref = parse(Float64, f[pos["leading_coefficient"]])
            for r in rows
                (r.pattern == nm && r.L == L) || continue
                ref_err = max(ref_err, abs(r.A_B / ref - 1))
                ref_n += 1
            end
        end
    end

    # ---- science summary
    println()
    corner = Dict{String,Float64}()
    for name in pattern_names
        sub = [r for r in rows if r.pattern == name]
        length(sub) >= 3 || continue
        L = Float64[r.L for r in sub]
        aA = Float64[r.A_A for r in sub]
        dD = Float64[r.D_A for r in sub]
        a, b, rms = lsq(L, aA)
        cl, bl, dl, crms = corner_fit(L, dD)
        corner[name] = cl
        @printf("%-18s A_A = %10.6f L %+10.4f (rms %.2e) | D_A: c_lnL = %8.5f (rms %.2e) | A_A/L@max = %9.5f\n",
            name, a, b, rms, cl, crms, aA[end] / L[end])
        @printf("%-18s   D_A = %s\n", "", join((@sprintf("%.4f", x) for x in dD), " "))
        # local diagnostics: is D_A logarithmic, and does A_A/L settle?
        loc = [(dD[k+1] - dD[k]) / log(L[k+1] / L[k]) for k in 1:length(L)-1]
        @printf("%-18s   local dD/dlnL = %s\n", "", join((@sprintf("%.4f", x) for x in loc), " "))
        @printf("%-18s   A_A/L         = %s\n", "", join((@sprintf("%.5f", x) for x in aA ./ L), " "))
    end

    # ---- strong-measurement tail test: c(beta) * exp(4 beta) -> c_lnL(D_A)
    #
    # The fixed-L expansion has control parameter t||X|| with t = e^{-2 beta}, so
    # the leading-order tail is approached from below with a relative deviation of
    # order t||X||.  ||X||_2 = sqrt(s_min^{-2} - 1) (SCALING_GUIDE eq. for X).
    println()
    tail_dev = NaN
    for name in pattern_names
        sub = [r for r in rows if r.pattern == name]
        isempty(sub) && continue
        smin = minimum(r.smin for r in sub)
        xnorm = sqrt(1 / smin^2 - 1)
        cl = get(corner, name, NaN)
        isfinite(cl) || continue
        @printf("%-18s s_min=%.4f  ||X||=%.3f  c_lnL(D_A)=%.4f\n", name, smin, xnorm, cl)
        haskey(FROZEN_C, name) || continue
        for (beta, c) in FROZEN_C[name]
            pred = c * exp(4beta)
            dev = abs(pred / cl - 1)
            @printf("    beta=%-5.2f  t||X||=%.3e  c(beta)e^{4beta}=%8.3f  ratio=%6.3f  dev=%5.2f%%\n",
                beta, exp(-2beta) * xnorm, pred, pred / cl, 100dev)
            if beta == FROZEN_C[name][end][1]
                tail_dev = isnan(tail_dev) ? dev : max(tail_dev, dev)
            end
        end
    end

    # ---- complement symmetry: uniform_B is the site-wise complement of
    # uniform_A, and the measured Rényi-2 entropy is invariant under taking the
    # complement of pattern and measurement set (SCALING_GUIDE §4.2), so A_R must
    # agree.  Independent of every external reference.
    comp_err = NaN
    let ua = [r for r in rows if r.pattern == "uniform_A"],
        ub = [r for r in rows if r.pattern == "uniform_B"]
        if !isempty(ua) && !isempty(ub)
            comp_err = 0.0
            for ra in ua, rb in ub
                ra.L == rb.L || continue
                for (x, y) in ((ra.A_A, rb.A_A), (ra.A_C, rb.A_C), (ra.A_B, rb.A_B))
                    comp_err = max(comp_err, abs(y / x - 1))
                end
            end
        end
    end

    # ---- persist
    csv = joinpath(HERE, "data", "strong_scaling_observations.csv")
    open(csv, "w") do io
        println(io, "pattern,m,L,path,smin,A_A,A_C,A_B,D_A,xcheck_rel_err")
        for r in rows
            println(io, join((r.pattern, r.m, r.L, r.path, r.smin, r.A_A, r.A_C,
                r.A_B, r.D_A, r.xcheck), ','))
        end
    end

    xchecks = [r.xcheck for r in rows if !isnan(r.xcheck)]
    lmax = isempty(rows) ? 0 : maximum(r.L for r in rows)
    println()
    @printf("METRIC aR_L_max=%d\n", lmax)
    @printf("METRIC n_points=%d\n", length(rows))
    @printf("METRIC ref_max_rel_err=%.3e\n", ref_err)
    @printf("METRIC ref_n=%d\n", ref_n)
    @printf("METRIC xcheck_max_rel_err=%.3e\n", isempty(xchecks) ? 0.0 : maximum(xchecks))
    @printf("METRIC oracle_max_rel_err=%.3e\n", oracle_err)
    @printf("METRIC complement_symmetry_err=%.3e\n", isnan(comp_err) ? 0.0 : comp_err)
    @printf("METRIC tail_rel_dev=%.3e\n", isnan(tail_dev) ? 0.0 : tail_dev)
    for (name, cl) in sort(collect(corner))
        @printf("METRIC corner_lnL_%s=%.5f\n", name, cl)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    t0 = time()
    main()
    @printf("METRIC pipeline_seconds=%.3f\n", time() - t0)
end
