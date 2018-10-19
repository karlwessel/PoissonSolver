
struct DiffusionPlan
    plan::PoissonPlan
end

DiffusionPlan(N::Int, dt::Real) = DiffusionPlan(N, N, N, 1/N, 1/N, 1/N, dt)
function DiffusionPlan(nx::Int, ny::Int, nz::Int,
        dx::Real, dy::Real, dz::Real, dt::Real)
    plan = PoissonPlan(nx, ny, nz, dx, dy, dz)
    diag = (2 .- dt*plan.diag) ./ (2 .+ dt*plan.diag) * prod(plan.size)
    DiffusionPlan(PoissonPlan(plan.planfw, plan.planbw, diag, plan.size, plan.temp))
end

execute!(p::DiffusionPlan, v::CuArray{Float32, 3}) = execute!(p.plan, v)

function execute!(p::DiffusionPlan, v::CuArray{Float32, 3}, steps::Int)
    for i in 1:steps
        execute!(p.plan, v)
    end
    v
end

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

function diffusion(ic::String, out::String, N::Int, steps::Int, dt::Real)
    println("Reading input from $(ic)...")
    v = readvolume(ic, N)
    println("Creating Plan for $(N)³...")
    plan = DiffusionPlan(N, dt)
    vgpu = cu(v)
    println("Executing $(steps) with Δt=$(dt)...")
    execute!(plan, vgpu, steps)
    v = collect(vgpu)
    println("Writing result to $(out)...")
    writevolume(out, v)
    println("done!")
end
