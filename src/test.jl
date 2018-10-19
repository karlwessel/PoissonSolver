using FFTW
using CuArrays
using LinearAlgebra
using CUDAnative
using FDMatrices: DCTI

include("poisson.jl")
include("diffusion.jl")

diffusion(ARGS[1], ARGS[2],
	parse(Int, ARGS[3]), parse(Int, ARGS[4]),
	parse(Float64, ARGS[5]), "dirichlet")
	
diffusion(ARGS[1], ARGS[2],
	parse(Int, ARGS[3]), parse(Int, ARGS[4]),
	parse(Float64, ARGS[5]))
