include("D:/JuliaDQMC/code/KAPDQMC/src/Measure/ai/strong_scaling.jl")
const FE=FermionExperiment
for (nm,occ) in (("uniform_A",BitVector([1,0])),("uniform_B",BitVector([0,1])))
  L=12; m=1
  model=FE.BlochModel(L,m); prep=FE.prepare_pattern(model,occ); base=FE.evolve(prep,Inf)
  kernel,eset,fset,smin=x_kernel(base)
  q=size(kernel,3)
  AA=aR_kernel(kernel,eset,fset,region_masks(model,L÷3,L))
  AC=aR_kernel(kernel,eset,fset,region_masks(model,L,L÷3))
  momx=0.0; momy=0.0; fro=0.0
  for dx in 0:q-1, dy in 0:q-1
    w=abs2(kernel[1,1,dx+1,dy+1])
    fro+=w
    momx+=w*min(min(dx,q-dx),L÷3)
    momy+=w*min(min(dy,q-dy),L÷3)
  end
  masks=region_masks(model,L÷3,L)
  ins=[dft2(x) for x in masks]; outs=[dft2(.!x) for x in masks]
  t1=0.0;t2=0.0
  for (ai,a) in enumerate(eset),(bi,b) in enumerate(fset)
    gh=dft2(abs2.(kernel[ai,bi,:,:]))
    t1+=corr_sum(gh,ins[a],outs[b],q)
    t2+=corr_sum(gh,outs[a],ins[b],q)
  end
  println("$nm L=$L  A_A=$AA  t1=$t1 t2=$t2  sum2=$(2*(t1+t2))")
  println("    fro=$fro momx=$momx 2L*momx=$(2*L*momx) A_A/L=$((AA)/L) ratio=$(AA/L/(2*momx))")
  println("    A_C=$AC 2L*momy=$(2*L*momy) ratio=$(AC/L/(2*momy))")
end
