# Small zero-mode comparison; reads saved L18 orbitals without changing them.
push!(LOAD_PATH, normpath(joinpath(@__DIR__, "..", "..", "..")))
using KAPDQMC, LinearAlgebra, SHA, TOML
BLAS.set_num_threads(1)

const OUT = @__DIR__
const BASE = normpath(joinpath(OUT,".."))
sha(path)=bytes2hex(sha256(read(path)))

function lognormalizer(a,beta)
    lw=zeros(a+1)
    for j in 1:a
        lw[j+1]=lw[j]+2log((a-j+1)/j)-4beta
    end
    top=maximum(lw)
    top+log(sum(exp.(lw.-top)))
end

function logweight(P,E,beta)
    filtered=copy(P)
    filtered[E,:].*=exp(-2beta)
    2sum(log,abs.(diag(qr(filtered).R)))-2sum(log,abs.(diag(qr(P).R)))
end

function main()
    L=18; a=L^2; N=2a; tol=1e-9
    ppath=joinpath(BASE,"data","local_orbitals_L18.bin")
    gpath=joinpath(BASE,"data","local_projector_L18.bin")
    oldsha=sha(ppath); gsha=sha(gpath)
    Pold=Matrix(reshape(reinterpret(Float64,read(ppath)),N,a))
    Gold_saved=Matrix(reshape(reinterpret(Float64,read(gpath)),N,N))
    H=nnK_Matrix("HoneyComb120",[L,L])
    eig=eigen(Symmetric(H))
    negative=findall(<(-1e-10),eig.values)
    zero=findall(x->abs(x)<1e-10,eig.values)
    @assert length(negative)==a-2 && length(zero)==4
    A=collect(1:2:N); B=collect(2:2:N)
    T=H[A,B]
    UAc=nullspace(T';atol=1e-10)
    UBc=nullspace(T;atol=1e-10)
    @assert size(UAc)==size(UBc)==(a,2)
    UA=zeros(N,2); UB=zeros(N,2)
    UA[A,:]=UAc; UB[B,:]=UBc
    Q0=(UA+UB)/sqrt(2)
    Pbalanced=hcat(eig.vectors[:,negative],Q0)
    E0=sum(eig.values[negative])
    diagnostics=Dict{String,Any}[]
    records=Any[]
    header=["state","L","target_pattern","suppressed_sublattice","beta","d_eigenvalues_near_one",
        "GE_min_eigenvalue","GE_max_eigenvalue","log_weight","log_probability","log10_probability",
        "balanced_analytic_log_probability","energy_error","orthogonality_error","G_AA_halfI_error","G_BB_halfI_error"]
    for (name,P) in (("saved_syevd",Pold),("balanced_zero_modes",Pbalanced))
        G=P*P'
        ee=abs(sum(P.*(H*P))-E0)
        orth=norm(P'*P-I)
        aa=norm(G[A,A]-.5I); bb=norm(G[B,B]-.5I)
        @assert ee<1e-9 && orth<1e-10
        if name=="balanced_zero_modes"
            @assert aa<1e-10 && bb<1e-10
        else
            @assert norm(G-Gold_saved)<1e-10
        end
        push!(diagnostics,Dict("state"=>name,"energy_error"=>ee,"orthogonality_error"=>orth,
            "G_AA_halfI_error"=>aa,"G_BB_halfI_error"=>bb))
        for (target,E) in (("uniform_A",B),("uniform_B",A))
            values=eigvals(Symmetric(G[E,E]))
            d=count(x->abs(x-1)<tol,values)
            for beta in (0.,1.,3.)
                lw=logweight(P,E,beta)
                lp=lw-lognormalizer(a,beta)
                analytic=a*log((1+exp(-4beta))/2)-lognormalizer(a,beta)
                name=="balanced_zero_modes" && @assert abs(lp-analytic)<1e-9
                push!(records,[name,L,target,target=="uniform_A" ? "B" : "A",beta,d,
                    minimum(values),maximum(values),lw,lp,lp/log(10),
                    name=="balanced_zero_modes" ? analytic : "",ee,orth,aa,bb])
            end
        end
    end
    csvpath=joinpath(OUT,"zero_mode_sensitivity.csv")
    open(csvpath,"w") do io
        println(io,join(header,','))
        for row in records;println(io,join(row,','));end
    end
    @assert sha(ppath)==oldsha && sha(gpath)==gsha
    receipt=Dict("status"=>"PASS","L"=>L,"zero_dimension"=>4,"zero_occupation"=>2,
        "eigenvalue_tolerance"=>tol,"target_convention"=>"target names denote occupied S; suppressed E is the opposite sublattice",
        "balanced_construction"=>"T=H[A,B]. U_A=null(T'), U_B=null(T), each arbitrary SVD-orthonormal real basis of dimension 2. Embed on pure sublattices and occupy Q0=(U_A+U_B)/sqrt2 together with all negative-energy orbitals. Pairing the two null bases is arbitrary and explicitly defines this example.",
        "scope"=>"Same Hamiltonian and half-filled ground energy; different occupied zero-mode subspace. This is not a rerun or correction of saved sampling trajectories.",
        "conclusion"=>"The exponent d and specific target probability may depend on the chosen degenerate Slater ground state. Zero overlap with one selected ground state does not mean the target product state cannot project into any ground state.",
        "balanced_formula"=>"G_AA=G_BB=I/2, so both uniform target weights equal ((1+exp(-4beta))/2)^a and d=0.",
        "states"=>diagnostics,"UA_H_residual"=>norm(H*UA),"UB_H_residual"=>norm(H*UB),
        "files"=>Dict("data/local_orbitals_L18.bin"=>oldsha,"data/local_projector_L18.bin"=>gsha,
            "analysis/zero_mode_sensitivity.jl"=>sha(@__FILE__),"analysis/zero_mode_sensitivity.csv"=>sha(csvpath)))
    open(joinpath(OUT,"zero_mode_sensitivity.toml"),"w") do io;TOML.print(io,receipt;sorted=true);end
    println("PASS: same-H zero-mode comparison, ",length(records)," probabilities")
    for row in records
        println(row[1]," ",row[3]," beta=",row[5]," d=",row[6]," log10p=",row[11])
    end
end
main()
