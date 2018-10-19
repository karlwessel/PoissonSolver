module DiffusionTests
using Test
using FDMatrices
using CuArrays
using PoissonSolver: DiffusionPlan, execute!, writevolume, readvolume
using PoissonSolver

@testset "DiffusionPlan run only tests" begin
    for BC in ["periodic", "dirichlet"]
        @testset "BC $BC" begin
            size = (8,8,8)
            dx = 1 ./ size
            dt = 1e-2
            g = rand(size...)
            plan = DiffusionPlan(size[1], dt, BC)
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
            plan = DiffusionPlan(size..., dx..., dt, BC)
            f2 = collect(execute!(plan, cu(g)))
            @test f1 == f2

            # test third constructor
            plan = DiffusionPlan(size, dx..., dt, BC)
            f2 = collect(execute!(plan, cu(g)))
            @test f1 == f2

            # test diffusion #1
            @test diffusion(g, dx..., 1, dt, BC) == f1

            # test diffusion #2
            @test diffusion(g, 1, dt, BC) == f1
        end
    end
end

@testset "Diffusion analytical tests" begin
    for size in [(8,8,8), (8,16,32)]
        @testset "Size: $size" begin
            dx = 1 ./ size
            dt = 1e-3
            t = 0.05
            steps = 50
            @test steps*dt ≈ t

            @testset "BC periodic" begin
                g = diff3dtest(size...)
                ref = diff3dtest(size..., t)
                maxref = maximum(abs.(ref))

                # make sure time progress is large enough for result to change
                @test maximum(abs.(g-ref)) / maxref > 1

                plan = DiffusionPlan(size..., dx..., dt)
                f = collect(execute!(plan, cu(g), steps))

                @test maximum(abs.(f-ref)) / maxref < 1.7e-2

                # make sure dirichlet BC don't give correct result
                plan = DiffusionPlan(size..., dx..., dt, "dirichlet")
                f = collect(execute!(plan, cu(g), steps))

                @test maximum(abs.(f-ref)) / maxref > 1
            end

            @testset "BC dirichlet" begin
                g = diff3dtestdirichlet(size...)
                ref = diff3dtestdirichlet(size..., t)
                maxref = maximum(abs.(ref))

                # make sure time progress is large enough for result to change
                @test maximum(abs.(g-ref)) / maxref > 1

                plan = DiffusionPlan(size..., dx..., dt, "dirichlet")
                f = collect(execute!(plan, cu(g), steps))

                @test maximum(abs.(f-ref)) / maxref < 1.7e-2

                # make sure periodic BC don't give correct result
                plan = DiffusionPlan(size..., dx..., dt, "periodic")
                f = collect(execute!(plan, cu(g), steps))

                @test maximum(abs.(f-ref)) / maxref > 1
            end
        end
    end
end

@testset "DiffusionPlan run slow tests" begin
    for BC in ["periodic", "dirichlet"]
        @testset "BC $BC" begin
            size = (8,8,8)
            dx = 1 ./ size
            dt = 1e-2
            g = rand(size...)
            plan = DiffusionPlan(size[1], dt, BC)
            ggpu = cu(g)
            fgpu = execute!(plan, ggpu)

            # test inplace op for single step execute!
            @test collect(ggpu) != g
            f1 = collect(fgpu)

            # test write and read
            filein = tempname()
            writevolume(filein, g)
            sleep(1)
            @test readvolume(filein, size...) ≈ g

            # test diffusion #2
            fileout = tempname()
            diffusion(filein, fileout, size[1], 1, dt, BC)
            sleep(1)
            @test readvolume(fileout, size...) ≈ f1
        end
    end
end
end
