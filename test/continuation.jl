using Motion 
using Motion.Continuation 
using LinearAlgebra
using Test

@testset "Layout" verbose=true begin 
    @testset "SingleShootingLayout" begin
        l = Continuation.SingleShootingLayout(6, true)
        @test Continuation.nx(l) == 6 
        @test Continuation.nvar(l) == 6 + 1

        z = collect(1:7)
        @test collect(1:6) == Continuation.unpack(l, z)[1]
        @test 7 == Continuation.unpack(l, z)[2]
    end;

    @testset "SingleShootingReducedLayout" begin 
        l = Continuation.SingleShootingReducedLayout(6, [1, 5], true)
        @test Continuation.nx(l) == 6 
        @test Continuation.nvar(l) == 2 + 1

        z = [1, 5, 7]
        @test [1, 0, 0, 0, 5, 0] == Continuation.unpack(l, z)[1]
        @test 7 == Continuation.unpack(l, z)[2]
    end;
end;

@testset "SingleShooting" verbose=true begin 

    @testset "SingleShootingLayout" begin 
        l = Continuation.SingleShootingLayout(6, true)
        flow = (x, T, λ) -> I(6) * x

        m = Continuation.SingleShooting(flow, l)
        @test Continuation.nx(m) == 6
        @test Continuation.nvar(m) == 6+1
        @test Continuation.layout(m) == l

        x0, T = Continuation.unpack(l, collect(1:7))
        @test m(x0, T, 0.0) == x0
    end

    @testset "SingleShootingReducedLayout" begin 
        l = Continuation.SingleShootingReducedLayout(6, [1, 5], true)
        flow = (x, T, λ) -> I(6) * x

        m = Continuation.SingleShooting(flow, l)
        @test Continuation.nx(m) == 6
        @test Continuation.nvar(m) == 2+1
        @test Continuation.layout(m) == l

        z = [1, 5, 7]
        x, T = Continuation.unpack(l, z)
        @test m(x, T, 0.0) == x
    end

    @testset "SingleShootingResidual (reduced layout)" begin 
        l = Continuation.SingleShootingReducedLayout(6, [1, 5], true)
        flow = (x, T, λ) -> I(6) * x

        m = Continuation.SingleShooting(flow, l)
        c = Continuation.Periodicity(l)
        r = Continuation.SingleShootingResidual(m, c)

        z = [1, 5, 7]
        @test sum(Continuation.residual(r, z, 0)) == 0
    end

end;
