module PoissonSolver
using LinearAlgebra
using FFTW
using CuArrays

export diffusion

include("poisson.jl")
include("diffusion.jl")

end # module
