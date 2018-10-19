module PoissonSolver
using LinearAlgebra
using FFTW
using CuArrays
using FDMatrices: DCTI

export diffusion

include("poisson.jl")
include("diffusion.jl")



end # module
