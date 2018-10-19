module PoissonSolver
using LinearAlgebra
using FFTW
using CuArrays

export diffusion

include("poisson.jl")
include("diffusion.jl")

"""
Read a N x M x L - array (volume) from the file with the passed
path.

Assumes data stored in C-compatible float32 binary format in
row-major-order.
"""
function readvolume(file::String, N::Int, M::Int, L::Int)
    v = Array{Float32}(undef, (L,M,N))
    read!(open(file), v)
    permutedims(v, (3,2,1))
end
readvolume(file::String, N::Int) = readvolume(file, N, N, N)

function writevolume(file::String, v::AbstractArray)
    write(open(file, "w"), permutedims(v, (3,2,1)))
end

function diffusion(v::AbstractArray{T, 3}, dx, dy, dz, steps, dt) where {T}
    println("Creating Plan for $(N)...")
    plan = DiffusionPlan(size(v), dx, dy, dz, dt)
    vgpu = cu(v)

    println("Executing $(steps) with Δt=$(dt)...")
    execute!(plan, vgpu, steps)
    collect(vgpu)
end
diffusion(v::AbstractArray{T, 3}, steps, dt) where {T} =
        diffusion(v, 1 ./ size(v), steps, dt)

function diffusion(ic::String, out::String, N::Int, steps::Int, dt::Real)
    println("Reading input from $(ic)...")
    v = readvolume(ic, N)

    v = diffusion(v, steps, dt)

    println("Writing result to $(out)...")
    writevolume(out, v)

    println("done!")
end

end # module
