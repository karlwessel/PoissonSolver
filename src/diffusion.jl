
struct DiffusionPlan
    plan::PoissonPlan
end

DiffusionPlan(N::Int, dt::Real) = DiffusionPlan(N, N, N, 1/N, 1/N, 1/N, dt)
DiffusionPlan(nx::Int, ny::Int, nz::Int, dx::Real, dy::Real, dz::Real, dt::Real) =
        DiffusionPlan((nx,ny,nz), dx, dy, dz, dt)
function DiffusionPlan(size::Tuple, dx::Real, dy::Real, dz::Real, dt::Real)
    diag = calcdiagpoisson(size, (dx,dy,dz))
    diag = (2 .+ dt*diag) ./ (2 .- dt*diag)
    plan = PoissonPlan(diag, size, dx, dy, dz)

    DiffusionPlan(plan)
end

execute!(p::DiffusionPlan, v::CuArray{Float32, 3}) = execute!(p.plan, v)

function execute!(p::DiffusionPlan, v::CuArray{Float32, 3}, steps::Int)
    for i in 1:steps
        execute!(p.plan, v)
    end
    v
end
