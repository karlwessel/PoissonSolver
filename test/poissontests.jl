module PoissonTests
using Test
using CuArrays
using PoissonSolver: PoissonPlan, calcdiagpoisson, execute!

@testset "PoissonPlan run only tests" begin
    for BC in ["periodic", "dirichlet"]
        @testset "BC $BC" begin
            size = (8,8,8)
            dx = 1 ./ size
            g = rand(size...)
            plan = PoissonPlan(8, BC)
            ggpu = cu(g)

            # test inplace execute!
            fgpu = execute!(plan, ggpu)
            @test collect(ggpu) != g
            f1 = collect(fgpu)
            @test f1 == collect(ggpu)

            # test second constructor
            plan = PoissonPlan(size..., dx..., BC)
            f2 = collect(execute!(plan, cu(g)))
            @test f1 == f2

            # test third constructor
            plan = PoissonPlan(size, dx..., BC)
            f2 = collect(execute!(plan, cu(g)))
            @test f1 == f2

            # test diag constructor
            diag = calcdiagpoisson(size, dx, BC)
            plan = PoissonPlan(diag, size, dx..., BC)
            f2 = collect(execute!(plan, cu(g)))
            @test f1 == f2
        end
    end
end

end
