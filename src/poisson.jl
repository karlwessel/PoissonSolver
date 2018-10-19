
function calcdiag(N::Int, dx, bc="periodic"; r2c=false)
    f = 1/(dx^2)/12
    if bc == "periodic"
        x = zeros(N)
        x[1:3] = f*[-30, 16, -1]
        x[end-1:end] = f*[-1, 16]
        r2c ? rfft(x) : fft(x)
    else
        x = zeros(N+2)
        x[1:3] = f*[-30, 16, -1]
        DCTI(x)[2:N+1]
    end
end

function calcdiagpoisson(dims::Tuple, dxs, bc="periodic")
    kx = calcdiag(dims[1], dxs[1], r2c=true)
    ky = calcdiag(dims[2], dxs[2])
    kz = calcdiag(dims[3], dxs[3], bc)

    ComplexF64[(x+y+z) for x in kx, y in ky, z in kz]
end

struct PoissonPlan
    planfw
    planbw
    diag::CuArray{Complex{Float32}, 3}
    size::Tuple{Int64,Int64,Int64}
    temp::CuArray{Complex{Float32}, 3}
end

PoissonPlan(N::Int) = PoissonPlan(N, N, N, 1/N, 1/N, 1/N)

function PoissonPlan(nx::Int, ny::Int, nz::Int, dx::Real, dy::Real, dz::Real)
    size = (nx,ny,nz)
    r2csize = (Int(nx/2+1), ny,nz)
    fw = plan_rfft(cu(Array{Float32}(undef, size)))
    temp = cu(Array{Complex{Float32}}(undef, r2csize))
    bw = plan_brfft(temp, nx)

    diag = cu(calcdiagpoisson(size, (dx,dy,dz)) * prod(size) )


    PoissonPlan(fw, bw, diag, size, temp)
end

function execute!(p::PoissonPlan, v::CuArray{Float32, 3})
    mul!(p.temp, p.planfw, v)
    p.temp .*= p.diag
    mul!(v, p.planbw, p.temp)
end
