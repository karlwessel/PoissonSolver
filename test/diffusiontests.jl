module DiffusionTests
using Test
using FDMatrices
using CuArrays
using PoissonSolver: DiffusionPlan, execute!

@testset "DiffusionPlan run only tests" begin
    size = (8,8,8)
    dx = 1 ./ size
    dt = 1e-2
    g = rand(size...)
    plan = DiffusionPlan(8, dt)
    ggpu = cu(g)
    fgpu = execute!(plan, ggpu)

    # test inplace op for single step execute!
    @test collect(ggpu) != g
    f1 = collect(fgpu)
    @test f1 == collect(ggpu)

    # test inplace op for multi step execute!
    ggpu = cu(g)
    fgpu = execute!(plan, ggpu, 1)
    @test collect(ggpu) != g
    f1 = collect(fgpu)
    @test collect(fgpu) == collect(ggpu)

    # test equality of multistep for one step
    @test collect(fgpu) == f1

    f10 = collect(execute!(plan, cu(g), 10))

    # test second constructor
    plan = DiffusionPlan(size..., dx..., dt)
    f2 = collect(execute!(plan, cu(g)))
    @test f1 == f2

    # test third constructor
    plan = DiffusionPlan(size, dx..., dt)
    f2 = collect(execute!(plan, cu(g)))
    @test f1 == f2
end

end
