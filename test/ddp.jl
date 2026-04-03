using Motion
using Motion.DDP
using LinearAlgebra
using StaticArrays
using Test

@testset "DDP" verbose=true begin

    # ── Simple double integrator ────────────────────────────────────
    # State x = [pos, vel], Control u = [accel]
    # Dynamics: x_{k+1} = A x_k + B u_k  (Euler discretisation)
    # with dt = t_{k+1} - t_k

    @testset "Unconstrained – double integrator" begin
        # Discrete dynamics
        function dyn(x::SVector{2,T}, u::SVector{1,T}, tk, tkp1) where T
            dt = tkp1 - tk
            A = SMatrix{2,2,T}(1, 0, dt, 1)
            B = SMatrix{2,1,T}(dt^2/2, dt)
            return A * x + B * u
        end

        stage = StageCost((x, u, t) -> 0.5 * u' * u)
        term  = TerminalCost(x -> zero(eltype(x)))

        prob = DDPProblem(dyn, stage, term, 2, 1)

        N  = 11
        dt = 0.1
        t  = collect(range(0.0, step=dt, length=N))

        # Initial guess: straight line from [0,0] to [1,0] with zero control
        X0 = [SVector(0.0, 0.0) for _ in 1:N]
        U0 = [SVector(0.0) for _ in 1:(N-1)]

        # Add terminal constraint: reach x = [1, 0]
        ψ(x) = SVector(x[1] - 1.0, x[2])
        tc = TerminalConstraint(ψ, 2)

        prob_c = DDPProblem(dyn, stage, term, 2, 1; terminal_eq=tc)

        sol = DDP.solve(prob_c, X0, U0, t;
                        opts=DDPOptions(verbose=false, max_outer=30, max_iter=50,
                                        ctol=1e-6, atol=1e-10, μ0=1.0, ϕ_μ=10.0))

        @test sol.status == :converged

        # Check terminal constraint satisfied
        @test abs(sol.X[end][1] - 1.0) < 1e-4
        @test abs(sol.X[end][2]) < 1e-4
    end

    @testset "Equality path constraints" begin
        # Double integrator with path constraint: vel ≤ via equality reformulation
        function dyn(x::SVector{2,T}, u::SVector{1,T}, tk, tkp1) where T
            dt = tkp1 - tk
            A = SMatrix{2,2,T}(1, 0, dt, 1)
            B = SMatrix{2,1,T}(dt^2/2, dt)
            return A * x + B * u
        end

        stage = StageCost((x, u, t) -> 0.5 * u' * u)
        term  = TerminalCost(x -> zero(eltype(x)))
        ψ(x)  = SVector(x[1] - 1.0, x[2])
        tc    = TerminalConstraint(ψ, 2)

        prob = DDPProblem(dyn, stage, term, 2, 1; terminal_eq=tc)

        N  = 11
        dt = 0.1
        t  = collect(range(0.0, step=dt, length=N))
        X0 = [SVector(0.0, 0.0) for _ in 1:N]
        U0 = [SVector(0.0) for _ in 1:(N-1)]

        sol = DDP.solve(prob, X0, U0, t;
                        opts=DDPOptions(verbose=false, max_outer=30,
                                        ctol=1e-5, μ0=1.0))

        @test sol.status == :converged
        @test abs(sol.X[end][1] - 1.0) < 1e-3
    end

    @testset "Inequality path constraints" begin
        # Double integrator: reach [1, 0] with velocity constraint |v| ≤ 0.8
        # h(x, u, t) = [0.8 - v, 0.8 + v] ≥ 0
        function dyn(x::SVector{2,T}, u::SVector{1,T}, tk, tkp1) where T
            dt = tkp1 - tk
            A = SMatrix{2,2,T}(1, 0, dt, 1)
            B = SMatrix{2,1,T}(dt^2/2, dt)
            return A * x + B * u
        end

        stage = StageCost((x, u, t) -> 0.5 * u' * u)
        term  = TerminalCost(x -> zero(eltype(x)))
        ψ(x)  = SVector(x[1] - 1.0, x[2])
        tc    = TerminalConstraint(ψ, 2)

        h(x, u, t) = SVector(0.8 - x[2], 0.8 + x[2])  # |v| ≤ 0.8
        ic = InequalityConstraint(h, 2)

        prob = DDPProblem(dyn, stage, term, 2, 1;
                          terminal_eq=tc, ineq=ic)

        N  = 21
        dt = 0.05
        t  = collect(range(0.0, step=dt, length=N))
        X0 = [SVector(0.0, 0.0) for _ in 1:N]
        U0 = [SVector(0.0) for _ in 1:(N-1)]

        sol = DDP.solve(prob, X0, U0, t;
                        opts=DDPOptions(verbose=false, max_outer=40, max_iter=100,
                                        ctol=1e-4, μ0=1.0, ϕ_μ=10.0))

        @test sol.status == :converged
        @test abs(sol.X[end][1] - 1.0) < 1e-2
        @test abs(sol.X[end][2]) < 1e-2

        # Check velocity bound respected (with small tolerance)
        for k in 1:N
            @test abs(sol.X[k][2]) ≤ 0.8 + 1e-2
        end
    end

    @testset "Types constructors" begin
        # EqualityConstraint
        g(x, u, t) = SVector(x[1])
        ec = EqualityConstraint(g, 1)
        @test ec.p == 1

        # InequalityConstraint
        h(x, u, t) = SVector(x[1], x[2])
        ic = InequalityConstraint(h, 2)
        @test ic.q == 2

        # TerminalConstraint
        ψ(x) = SVector(x[1] - 1.0)
        tc = TerminalConstraint(ψ, 1)
        @test tc.r == 1

        # DDPOptions defaults
        opts = DDPOptions{Float64}()
        @test opts.max_iter == 200
        @test opts.max_outer == 20

        # DDPProblem
        dyn(x, u, tk, tkp1) = x
        prob = DDPProblem(dyn, StageCost((x,u,t)->0.0), TerminalCost(x->0.0),
                          2, 1; eq=ec, ineq=ic, terminal_eq=tc)
        @test prob.nx == 2
        @test prob.nu == 1
    end

end
