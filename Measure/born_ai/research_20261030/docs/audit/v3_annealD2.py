import json
res = json.load(open("d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030/docs/audit/v3_annealD.json"))
print(len(res))
for r in sorted(res, key=lambda r: (r[0], r[2])):
    print("L=%2d beta=%.4f z=%9.3f kappa=%.4g D_ann=%9.4f D_bar=%.5f ratio=%10.2f" % tuple(r))
