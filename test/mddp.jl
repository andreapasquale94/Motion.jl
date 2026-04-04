using Motion
using Motion.DDP: StageCost, TerminalCost, TerminalConstraint,
                  InequalityConstraint, EqualityConstraint
using Motion.MDDP
using LinearAlgebra
using StaticArrays
using Test

# ── Shared test dynamics ────────────────────────────────────────────
function double_integrator(x::SVector{2,T}, u::SVector{1,T}, tk, tkp1) where T
    dt = tkp1 - tk
    A = SMatrix{2,2,T}(1, 0, dt, 1)
    B = SMatrix{2,1,T}(dt^2/2, dt)
    return A * x + B * u
end

@testset "MDDP" verbose=true begin

    @testset "Single leg (M=1) – equivalent to standard DDP" begin
        stage = StageCost((x, u, t) -> 0.5 * u' * u)
        term  = TerminalCost(x -> zero(eltype(x)))
        ψ(x)  = SVector(x[1] - 1.0, x[2])
        tc    = TerminalConstraint(ψ, 2)

        prob = MDDPProblem(double_integrator, stage, term, 2, 1; terminal_eq=tc)

        N  = 11
        t  = collect(range(0.0, step=0.1, length=N))
        X0 = [SVector(0.0, 0.0) for _ in 1:N]
        U0 = [SVector(0.0) for _ in 1:(N-1)]

        sol = MDDP.solve(prob, X0, U0, t, 1;
                        opts=MDDPOptions(verbose=false, max_outer=30,
                                         max_ddp_iter=50, ctol=1e-5,
                                         μ0=1.0, ϕ_μ=10.0))

        @test sol.status == :converged
        @test abs(sol.X[end][1] - 1.0) < 1e-3
        @test abs(sol.X[end][2]) < 1e-3
    end

    @testset "Two legs (M=2) – terminal constraint" begin
        stage = StageCost((x, u, t) -> 0.5 * u' * u)
        term  = TerminalCost(x -> zero(eltype(x)))
        ψ(x)  = SVector(x[1] - 1.0, x[2])
        tc    = TerminalConstraint(ψ, 2)

        prob = MDDPProblem(double_integrator, stage, term, 2, 1; terminal_eq=tc)

        N  = 21
        t  = collect(range(0.0, step=0.05, length=N))
        X0 = [SVector(0.0, 0.0) for _ in 1:N]
        U0 = [SVector(0.0) for _ in 1:(N-1)]

        sol = MDDP.solve(prob, X0, U0, t, 2;
                        opts=MDDPOptions(verbose=false, max_outer=40,
                                         max_ddp_iter=50, max_node_iter=10,
                                         ctol=1e-4, dtol=1e-4,
                                         μ0=1.0, ϕ_μ=10.0))

        @test sol.status == :converged
        @test abs(sol.X[end][1] - 1.0) < 1e-2
        @test abs(sol.X[end][2]) < 1e-2
    end

    @testset "Three legs (M=3) – inequality constraints" begin
        stage = StageCost((x, u, t) -> 0.5 * u' * u)
        term  = TerminalCost(x -> zero(eltype(x)))
        ψ(x)  = SVector(x[1] - 1.0, x[2])
        tc    = TerminalConstraint(ψ, 2)

        h(x, u, t) = SVector(0.8 - x[2], 0.8 + x[2])
        ic = InequalityConstraint(h, 2)

        prob = MDDPProblem(double_integrator, stage, term, 2, 1;
                           terminal_eq=tc, ineq=ic)

        N  = 31
        t  = collect(range(0.0, step=0.05, length=N))
        X0 = [SVector(0.0, 0.0) for _ in 1:N]
        U0 = [SVector(0.0) for _ in 1:(N-1)]

        sol = MDDP.solve(prob, X0, U0, t, 3;
                        opts=MDDPOptions(verbose=false, max_outer=50,
                                         max_ddp_iter=80, max_node_iter=15,
                                         ctol=1e-3, dtol=1e-3,
                                         μ0=1.0, ϕ_μ=10.0))

        @test sol.status == :converged
        @test abs(sol.X[end][1] - 1.0) < 5e-2
        @test abs(sol.X[end][2]) < 5e-2
    end

    @testset "Types constructors" begin
        dyn(x, u, tk, tkp1) = x
        stage = StageCost((x, u, t) -> 0.0)
        term  = TerminalCost(x -> 0.0)

        prob = MDDPProblem(dyn, stage, term, 2, 1)
        @test prob.nx == 2
        @test prob.nu == 1

        opts = MDDPOptions{Float64}()
        @test opts.method == :iLQR
        @test opts.max_ddp_iter == 100
        @test opts.max_outer == 30

        # Leg construction
        X = [SVector(0.0, 0.0), SVector(1.0, 0.0)]
        U = [SVector(0.5)]
        t = [0.0, 1.0]
        leg = MDDP.Leg(X, U, t)
        @test length(leg.X) == 2
        @test length(leg.U) == 1
    end

    @testset "Leg splitting" begin
        N = 11
        X0 = [SVector(Float64(i), 0.0) for i in 1:N]
        U0 = [SVector(0.0) for _ in 1:(N-1)]
        t  = collect(range(0.0, step=0.1, length=N))

        legs = MDDP._split_into_legs(X0, U0, t, 2)
        @test length(legs) == 2
        @test legs[1].t[1] == t[1]
        @test legs[2].t[end] == t[end]
        @test length(legs[1].X) + length(legs[2].X) - 1 ≥ N - 1
    end

end
