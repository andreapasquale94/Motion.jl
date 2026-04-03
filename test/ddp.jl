using Motion
using Motion.DDP
using LinearAlgebra
using StaticArrays
using Test

# ── Shared test dynamics ────────────────────────────────────────────
# Double integrator: x = [pos, vel], u = [accel]
function double_integrator(x::SVector{2,T}, u::SVector{1,T}, tk, tkp1) where T
    dt = tkp1 - tk
    A = SMatrix{2,2,T}(1, 0, dt, 1)
    B = SMatrix{2,1,T}(dt^2/2, dt)
    return A * x + B * u
end

@testset "DDP" verbose=true begin

    @testset "iLQR – terminal constrained double integrator" begin
        stage = StageCost((x, u, t) -> 0.5 * u' * u)
        term  = TerminalCost(x -> zero(eltype(x)))
        ψ(x)  = SVector(x[1] - 1.0, x[2])
        tc    = TerminalConstraint(ψ, 2)

        prob = DDPProblem(double_integrator, stage, term, 2, 1; terminal_eq=tc)

        N  = 11
        t  = collect(range(0.0, step=0.1, length=N))
        X0 = [SVector(0.0, 0.0) for _ in 1:N]
        U0 = [SVector(0.0) for _ in 1:(N-1)]

        sol = DDP.solve(prob, X0, U0, t;
                        opts=DDPOptions(method=:iLQR, verbose=false,
                                        max_outer=30, max_iter=50,
                                        ctol=1e-6, atol=1e-10, μ0=1.0, ϕ_μ=10.0))

        @test sol.status == :converged
        @test abs(sol.X[end][1] - 1.0) < 1e-4
        @test abs(sol.X[end][2]) < 1e-4
    end

    @testset "Full DDP – terminal constrained double integrator" begin
        stage = StageCost((x, u, t) -> 0.5 * u' * u)
        term  = TerminalCost(x -> zero(eltype(x)))
        ψ(x)  = SVector(x[1] - 1.0, x[2])
        tc    = TerminalConstraint(ψ, 2)

        prob = DDPProblem(double_integrator, stage, term, 2, 1; terminal_eq=tc)

        N  = 11
        t  = collect(range(0.0, step=0.1, length=N))
        X0 = [SVector(0.0, 0.0) for _ in 1:N]
        U0 = [SVector(0.0) for _ in 1:(N-1)]

        sol = DDP.solve(prob, X0, U0, t;
                        opts=DDPOptions(method=:DDP, verbose=false,
                                        max_outer=30, max_iter=50,
                                        ctol=1e-6, atol=1e-10, μ0=1.0, ϕ_μ=10.0))

        @test sol.status == :converged
        @test abs(sol.X[end][1] - 1.0) < 1e-4
        @test abs(sol.X[end][2]) < 1e-4
    end

    @testset "iLQR – inequality path constraints" begin
        stage = StageCost((x, u, t) -> 0.5 * u' * u)
        term  = TerminalCost(x -> zero(eltype(x)))
        ψ(x)  = SVector(x[1] - 1.0, x[2])
        tc    = TerminalConstraint(ψ, 2)

        h(x, u, t) = SVector(0.8 - x[2], 0.8 + x[2])  # |v| ≤ 0.8
        ic = InequalityConstraint(h, 2)

        prob = DDPProblem(double_integrator, stage, term, 2, 1;
                          terminal_eq=tc, ineq=ic)

        N  = 21
        t  = collect(range(0.0, step=0.05, length=N))
        X0 = [SVector(0.0, 0.0) for _ in 1:N]
        U0 = [SVector(0.0) for _ in 1:(N-1)]

        sol = DDP.solve(prob, X0, U0, t;
                        opts=DDPOptions(method=:iLQR, verbose=false,
                                        max_outer=40, max_iter=100,
                                        ctol=1e-4, μ0=1.0, ϕ_μ=10.0))

        @test sol.status == :converged
        @test abs(sol.X[end][1] - 1.0) < 1e-2
        @test abs(sol.X[end][2]) < 1e-2

        for k in 1:N
            @test abs(sol.X[k][2]) ≤ 0.8 + 1e-2
        end
    end

    @testset "Types constructors" begin
        g(x, u, t) = SVector(x[1])
        ec = EqualityConstraint(g, 1)
        @test ec.p == 1

        h(x, u, t) = SVector(x[1], x[2])
        ic = InequalityConstraint(h, 2)
        @test ic.q == 2

        ψ(x) = SVector(x[1] - 1.0)
        tc = TerminalConstraint(ψ, 1)
        @test tc.r == 1

        # DDPOptions defaults
        opts = DDPOptions{Float64}()
        @test opts.method == :iLQR
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
