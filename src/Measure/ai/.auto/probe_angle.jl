include("D:/JuliaDQMC/code/KAPDQMC/src/Measure/ai/strong_scaling.jl")
const FE=FermionExperiment
# identity: sigma_max(X)^2 == 1/lambda_min - 1,  lambda_min = min_k sigma_min(Q_F(k))^2
# and the principal-angle reading  ||X|| = tan(theta_max)
for (nm,occ,m) in (("uniform_A",BitVector([1,0]),1),("one_B_2",special_pattern(2,[(2,2)]),2))
  L = 3m*2
  model=FE.BlochModel(L,m); prep=FE.prepare_pattern(model,occ); base=FE.evolve(prep,Inf)
  eset,fset=findall(.!occ),findall(occ)
  smaxX=0.0; lam_min=Inf
  for k in 1:L^2÷m^2
    u=base.orbitals[k]; qf=u[fset,:]
    s=svdvals(qf); lam_min=min(lam_min,minimum(s)^2)
    Xk=u[eset,:]/qf
    smaxX=max(smaxX, maximum(svdvals(Xk)))
  end
  pred=sqrt(1/lam_min-1)
  println("$nm m=$m L=$L: sigma_max(X)=$(round(smaxX,digits=10))  sqrt(1/lam_min-1)=$(round(pred,digits=10))  ratio=$(round(smaxX/pred,digits=12))")
  println("    lambda_min=$(round(lam_min,digits=10))  theta_max=$(round(rad2deg(acos(sqrt(lam_min))),digits=4)) deg  tan(theta_max)=$(round(tan(acos(sqrt(lam_min))),digits=8))")
end
