
struct DiffusionPlan
    plan::PoissonPlan
end

DiffusionPlan(N::Int, dt::Real, BC = "periodic") =
        DiffusionPlan(N, N, N, 1/N, 1/N, 1/N, dt, BC)
DiffusionPlan(nx::Int, ny::Int, nz::Int, dx::Real, dy::Real, dz::Real, dt::Real, BC = "periodic") =
        DiffusionPlan((nx,ny,nz), dx, dy, dz, dt, BC)
function DiffusionPlan(size::Tuple, dx::Real, dy::Real, dz::Real, dt::Real, BC = "periodic")
    diag = calcdiagpoisson(size, (dx,dy,dz), BC)
    diag = (2 .+ dt*diag) ./ (2 .- dt*diag)
    plan = PoissonPlan(diag, size, dx, dy, dz, BC)

    DiffusionPlan(plan)
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

"""
Write a N x M x L - array (volume) to the file with the passed
path.

Data is stored in C-compatible float32 binary format in
row-major-order.
"""
function writevolume(file::String, v::AbstractArray)
    write(open(file, "w"), permutedims(Array{Float32}(v), (3,2,1)))
end

"""
Execute the passed amount of time steps for the passed initial condition
and sample distance.

v - initial condition (IC) of the 3d diffusion equation to solve
dx, dy, dz - the sampling distance of the IC
step - the number of time steps to forward the IC
dt - the size of one time step
"""
function diffusion(v::AbstractArray{T, 3}, dx, dy, dz, steps, dt, BC = "periodic") where {T}
    println("Creating Plan for $(size(v))...")
    plan = DiffusionPlan(size(v), dx, dy, dz, dt, BC)
    vgpu = cu(v)

    println("Executing $(steps) with Δt=$(dt)...")
    execute!(plan, vgpu, steps)
    collect(vgpu)
end
"""
Execute the passed amount of time steps for the passed initial condition
and sample distance.

Assumes the domain of the input data is zero to one and therefore the
sampling distance is 1 / size(v, i) in dimension i.

v - initial condition (IC) of the 3d diffusion equation to solve
step - the number of time steps to forward the IC
dt - the size of one time step
"""
diffusion(v::AbstractArray{T, 3}, steps, dt, BC = "periodic") where {T} =
        diffusion(v, (1 ./ size(v))..., steps, dt, BC)

"""
Execute the passed amount of time steps for the passed initial condition
and sample distance.

Assumes the domain of the input data is zero to one and therefore the
sampling distance is 1 / size(v, i) in dimension i.

ic - path to the file containing the initial condition (IC) of the 3d diffusion
    equation to solve. The data should be stored in c compatible binary format
    of float32 values.
out - path to the file where the result should be stored to. Data format of the
    result will be the same as for the input IC file.
step - the number of time steps to forward the IC
dt - the size of one time step
"""
function diffusion(ic::String, out::String, N::Int, steps::Int, dt::Real,
        BC = "periodic")
    println("Reading input from $(ic)...")
    v = readvolume(ic, N)

    v = diffusion(v, steps, dt, BC)

    println("Writing result to $(out)...")
    writevolume(out, v)

    println("done!")
end
