using Pkg
Pkg.activate(pwd())
Pkg.add("Printf")
Pkg.resolve()
Pkg.instantiate()
Pkg.precompile()
println("done")
