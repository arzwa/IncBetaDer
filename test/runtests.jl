using Test, IncBetaDer, FiniteDifferences, SpecialFunctions, ForwardDiff

import IncBetaDer: beta_inc_

@testset "Incomplete Beta derivatives" begin
    xs = [(.1, 1.5, 11.), (.5, 1.5, 11.), (.8,   .3, .7), (.4,  12.,   4.),
          (.01, .2,  .5), (.001, .2, .3), (.999, 2., .2), (.2, 100., 200.)]

    @testset "Derivative of K" begin
        abs = [0.1, 1., 10., 100]
        for a in abs, b in abs
            r = rand()
            y = grad(central_fdm(5,1), x->IncBetaDer.Kfun(r, x...), [a, b])[1]
            z = IncBetaDer.dK_dpdq(r, a, b)
            @test y[1] ≈ z[1] atol=1e-7
            @test y[2] ≈ z[2] atol=1e-7
        end
    end

    @testset "I(x;a,b) continued fraction" begin
        for (x, p, q) in xs 
            y1 = IncBetaDer.recursive_cf(x, p, q, 100)
            y2 = beta_inc(p, q, x)[1]
            @info x, p, q
            @test y1 ≈ y2 atol=1e-5 
        end
    end

    @testset "Gradient, direct" begin
        for (x, p, q) in xs
            y1 = beta_inc(p, q, x)[1]
            y2 = beta_inc_grad(p, q, x)
            @test y2[1] ≈ y1 atol=1e-5
            g = grad(central_fdm(5,1), y->beta_inc(y...)[1], [p, q, x])[1]
            f = ForwardDiff.gradient(x->beta_inc_(x...), [p, q, x])
            for i=1:3
                @test y2[i+1] ≈ g[i] atol=1e-5
                @test y2[i+1] ≈ f[i] atol=1e-5
            end
        end
    end

    @testset "Gradient, ForwardDiff" begin
        for (x, p, q) in xs
            g = grad(central_fdm(5,1), y->beta_inc_inv(y...)[1], [p, q, x])[1]
            f = ForwardDiff.gradient(x->beta_inc_inv_(x...), [p, q, x])
            for i=1:3
                @test f[i] ≈ g[i] atol=1e-5
            end
        end
    end

    @testset "Gradients, ForwardDiff (2)" begin 
        for i=1:100
            x = rand()
            p, q = -10log.(rand(2))
            f = ForwardDiff.gradient(x->beta_inc_inv_(x...), [p, q, x])
            @test all(isfinite.(f))
        end
    end

end
