
function calcdiag(N::Int, dx, bc="periodic"; r2c=false)
    f = 1/(dx^2)/12
    if bc == "periodic"
        x = zeros(N)
        x[1:3] = f*[-30, 16, -1]
        x[end-1:end] = f*[-1, 16]
        r2c ? rfft(x) : fft(x)
    else
        x = zeros(N+1)
        x[1:3] = f*[-30, 16, -1]
        DCTI(x)[1:N]
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
    dirichlet::Bool
end

PoissonPlan(N::Int, BC = "periodic") = PoissonPlan((N, N, N), 1/N, 1/N, 1/N, BC)
PoissonPlan(nx::Int, ny::Int, nz::Int, dx::Real, dy::Real, dz::Real, BC = "periodic") =
        PoissonPlan((nx, ny, nz), dx, dy, dz, BC)
PoissonPlan(size::Tuple, dx::Real, dy::Real, dz::Real, BC = "periodic") =
        PoissonPlan(calcdiagpoisson(size, (dx,dy,dz), BC), size, dx, dy, dz, BC)
function PoissonPlan(diag, size, dx::Real, dy::Real, dz::Real, BC = "periodic")
    @assert BC in ["periodic", "dirichlet"]

    fftsize = (size[1], size[2], BC == "periodic" ? size[3] : size[3] * 2)
    r2csize = (Int(fftsize[1]/2+1), fftsize[2], fftsize[3])

    fw = plan_rfft(cu(Array{Float32}(undef, fftsize)))
    temp = cu(Array{Complex{Float32}}(undef, r2csize))
    bw = plan_brfft(temp, size[1])

    scale = prod(size)
    if BC == "dirichlet" scale /= 2 end

    diag = cu(diag / scale )

    PoissonPlan(fw, bw, diag, size, temp, BC != "periodic")
end

function execute!(p::PoissonPlan, v::CuArray{Float32, 3})
    NZ = p.size[3]
    if p.dirichlet
        vd = CuArray{Float32}(undef, (p.size[1], p.size[2], NZ*2))
        vd[:,:,1:NZ] .= v
        vd[:,:,NZ+1:end] .= 0
    else
        vd = v
    end
    mul!(p.temp, p.planfw, vd)
    if p.dirichlet
        p.temp[:,:, 2:NZ] .= (p.temp[:,:,2:NZ]
            .- p.temp[:,:,end:-1:NZ+2]) .* p.diag[:,:,2:NZ]
        p.temp[:,:,end:-1:NZ+2] .= -p.temp[:,:, 2:NZ]
        p.temp[:,:, 1] .= 0
        p.temp[:,:,NZ+1] .= 0
    else
        p.temp .*= p.diag
    end
    mul!(vd, p.planbw, p.temp)
    if p.dirichlet
        v .= vd[:,:,1:NZ]
    end
    v
end
