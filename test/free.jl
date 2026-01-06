push!(LOAD_PATH, "D:\\JuliaDQMC\\code\\KAPDQMC\\src\\public\\")

using Geometry



site = [3, 3]

nnn = nnn2idx("HoneyComb120", site, 9)

K=nnnK_Matrix("HoneyComb120", site)
