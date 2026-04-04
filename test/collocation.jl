using Motion
using Motion.Collocation
using LinearAlgebra
using StaticArrays
using Test

# ── Test dynamics: simple harmonic oscillator ẋ = [v, -x] ─────────
function sho_dynamics(x::SVector{2,T}, u::SVector{1,T}, t) where T
    return SVector{2,T}(x[2] + u[1], -x[1])
end

# ── Test dynamics: double integrator ẋ = [v, u] ───────────────────
function di_dynamics(x::SVector{2,T}, u::SVector{1,T}, t) where T
    return SVector{2,T}(x[2], u[1])
end

@testset "Collocation" verbose=true begin

    @testset "Compressed layout" begin
        N, nx, nu = 5, 2, 1
        it0, idt, iX, iU = indexes(Val(N), Val(nx), Val(nu))

        @test it0 == 1
        @test length(idt) == N - 1
        @test size(iX) == (nx, N)
        @test size(iU) == (nu, N)

        # Total size
        total = N + nx * N + nu * N
        @test maximum(vec(iU)) == total

        # No overlaps
        all_idx = vcat([it0], collect(idt), vec(iX), vec(iU))
        @test length(unique(all_idx)) == total
    end

    @testset "Separated layout" begin
        N, nx, nu = 5, 2, 1
        it0, idt, iX, iU, iXm, iUm = indexes_separated(Val(N), Val(nx), Val(nu))

        @test size(iXm) == (nx, N - 1)
        @test size(iUm) == (nu, N - 1)

        total = N + nx * (2N - 1) + nu * (2N - 1)
        @test maximum(vec(iUm)) == total

        all_idx = vcat([it0], collect(idt), vec(iX), vec(iU), vec(iXm), vec(iUm))
        @test length(unique(all_idx)) == total
    end

    @testset "Variables unpacking" begin
        N, nx, nu = 4, 2, 1
        total = N + nx * N + nu * N
        vars = collect(1.0:total)

        t0, dt, X, U, t = variables(vars, Val(N), Val(nx), Val(nu))
        @test t0 == 1.0
        @test length(dt) == N - 1
        @test size(X) == (nx, N)
        @test size(U) == (nu, N)
        @test t[1] == t0
        @test t[2] ≈ t0 + dt[1]
    end

    @testset "Variables separated unpacking" begin
        N, nx, nu = 4, 2, 1
        total = N + nx * (2N - 1) + nu * (2N - 1)
        vars = collect(1.0:total)

        t0, dt, X, U, Xm, Um, t = variables_separated(vars, Val(N), Val(nx), Val(nu))
        @test size(Xm) == (nx, N - 1)
        @test size(Um) == (nu, N - 1)
    end

    @testset "Compressed defects – exact trajectory" begin
        # Double integrator: x(t) = [t²/2, t] with u=1 satisfies ẋ = [v, 1]
        N, nx, nu = 6, 2, 1
        tf = 1.0
        t_nodes = collect(range(0.0, tf, length=N))

        # Build exact trajectory
        X_exact = [SVector(tn^2 / 2, tn) for tn in t_nodes]
        U_exact = [SVector(1.0) for _ in 1:N]

        # Pack into decision vector
        t0 = t_nodes[1]
        dts = diff(t_nodes)
        total = N + nx * N + nu * N
        vars = zeros(total)
        it0, idt, iX, iU = indexes(Val(N), Val(nx), Val(nu))
        vars[it0] = t0
        vars[idt] .= dts
        for k in 1:N
            vars[iX[:, k]] .= X_exact[k]
            vars[iU[:, k]] .= U_exact[k]
        end

        d = defects(vars, di_dynamics, Val(N), Val(nx), Val(nu), Val(:HermiteSimpson))
        @test maximum(abs, d) < 1e-12
    end

    @testset "Separated defects – exact trajectory" begin
        N, nx, nu = 6, 2, 1
        tf = 1.0
        t_nodes = collect(range(0.0, tf, length=N))

        X_exact = [SVector(tn^2 / 2, tn) for tn in t_nodes]
        U_exact = [SVector(1.0) for _ in 1:N]

        t0 = t_nodes[1]
        dts = diff(t_nodes)

        # Midpoint values
        Xm_exact = [SVector(((t_nodes[k] + t_nodes[k+1]) / 2)^2 / 2,
                            (t_nodes[k] + t_nodes[k+1]) / 2) for k in 1:(N-1)]
        Um_exact = [SVector(1.0) for _ in 1:(N-1)]

        total = N + nx * (2N - 1) + nu * (2N - 1)
        vars = zeros(total)
        it0, idt, iX, iU, iXm, iUm = indexes_separated(Val(N), Val(nx), Val(nu))
        vars[it0] = t0
        vars[idt] .= dts
        for k in 1:N
            vars[iX[:, k]] .= X_exact[k]
            vars[iU[:, k]] .= U_exact[k]
        end
        for k in 1:(N-1)
            vars[iXm[:, k]] .= Xm_exact[k]
            vars[iUm[:, k]] .= Um_exact[k]
        end

        d = defects(vars, di_dynamics, Val(N), Val(nx), Val(nu),
                    Val(:HermiteSimpsonSeparated))
        @test maximum(abs, d) < 1e-12
    end

    @testset "Compressed defects – non-zero for wrong trajectory" begin
        N, nx, nu = 4, 2, 1
        total = N + nx * N + nu * N
        vars = zeros(total)
        it0, idt, iX, iU = indexes(Val(N), Val(nx), Val(nu))
        vars[it0] = 0.0
        vars[idt] .= 0.5
        # Zero states, unit control → defects should be non-zero
        for k in 1:N
            vars[iU[:, k]] .= [1.0]
        end

        d = defects(vars, di_dynamics, Val(N), Val(nx), Val(nu), Val(:HermiteSimpson))
        @test maximum(abs, d) > 0.01
    end

    @testset "Objective – FUEL (compressed)" begin
        N, nx, nu = 4, 2, 1
        total = N + nx * N + nu * N
        vars = zeros(total)
        it0, idt, iX, iU = indexes(Val(N), Val(nx), Val(nu))
        vars[it0] = 0.0
        vars[idt] .= 1.0
        for k in 1:N
            vars[iU[:, k]] .= [2.0]
        end

        J_fuel = objective(vars, Val(N), Val(nx), Val(nu), Val(:FUEL))
        # Simpson quadrature of constant ‖u‖=2 over 3 segments of length 1 = 6
        @test J_fuel ≈ 6.0 atol=1e-8

        J_energy = objective(vars, Val(N), Val(nx), Val(nu), Val(:ENERGY))
        # Simpson of constant ‖u‖²=4 over 3 segments of length 1 = 12
        @test J_energy ≈ 12.0 atol=1e-8

        J_time = objective(vars, Val(N), Val(nx), Val(nu), Val(:TIME))
        @test J_time ≈ 3.0
    end

    @testset "Objective – separated" begin
        N, nx, nu = 4, 2, 1
        total = N + nx * (2N - 1) + nu * (2N - 1)
        vars = zeros(total)
        it0, idt, iX, iU, iXm, iUm = indexes_separated(Val(N), Val(nx), Val(nu))
        vars[it0] = 0.0
        vars[idt] .= 1.0
        for k in 1:N
            vars[iU[:, k]] .= [2.0]
        end
        for k in 1:(N-1)
            vars[iUm[:, k]] .= [2.0]
        end

        J_fuel = objective_separated(vars, Val(N), Val(nx), Val(nu), Val(:FUEL))
        @test J_fuel ≈ 6.0 atol=1e-8

        J_energy = objective_separated(vars, Val(N), Val(nx), Val(nu), Val(:ENERGY))
        @test J_energy ≈ 12.0 atol=1e-8
    end

    @testset "Mesh error – exact polynomial" begin
        # For a quadratic trajectory, Hermite-Simpson is exact → error ≈ 0
        N, nx, nu = 6, 2, 1
        tf = 1.0
        t_nodes = collect(range(0.0, tf, length=N))
        X = [SVector(tn^2 / 2, tn) for tn in t_nodes]
        U = [SVector(1.0) for _ in 1:N]

        errs = mesh_error(X, U, t_nodes, di_dynamics)
        @test all(e -> e < 1e-12, errs)
    end

    @testset "Mesh error – non-polynomial detects error" begin
        # SHO is not polynomial → error should be non-zero on coarse mesh
        N = 4
        tf = π
        t_nodes = collect(range(0.0, tf, length=N))
        # Approximate SHO trajectory with zero control
        X = [SVector(cos(tn), -sin(tn)) for tn in t_nodes]
        U = [SVector(0.0) for _ in 1:N]

        errs = mesh_error(X, U, t_nodes, sho_dynamics)
        @test any(e -> e > 1e-4, errs)
    end

    @testset "Mesh refinement – bisects high-error segments" begin
        N = 4
        tf = π
        t_nodes = collect(range(0.0, tf, length=N))
        X = [SVector(cos(tn), -sin(tn)) for tn in t_nodes]
        U = [SVector(0.0) for _ in 1:N]

        X_new, U_new, t_new, refined = refine_mesh(X, U, t_nodes, sho_dynamics;
                                                     tol=1e-4)
        @test refined
        @test length(X_new) > N
        @test length(U_new) == length(X_new)
        @test length(t_new) == length(X_new)

        # Time ordering preserved
        @test issorted(t_new)
        # Endpoints preserved
        @test t_new[1] == t_nodes[1]
        @test t_new[end] == t_nodes[end]
    end

    @testset "Mesh refinement – no refinement needed" begin
        # Exact polynomial → no refinement
        N = 6
        tf = 1.0
        t_nodes = collect(range(0.0, tf, length=N))
        X = [SVector(tn^2 / 2, tn) for tn in t_nodes]
        U = [SVector(1.0) for _ in 1:N]

        _, _, _, refined = refine_mesh(X, U, t_nodes, di_dynamics; tol=1e-6)
        @test !refined
    end

    @testset "Mesh refinement – iterated refinement reduces error" begin
        N = 4
        tf = π
        t_nodes = collect(range(0.0, tf, length=N))
        X = [SVector(cos(tn), -sin(tn)) for tn in t_nodes]
        U = [SVector(0.0) for _ in 1:N]

        max_err_before = maximum(mesh_error(X, U, t_nodes, sho_dynamics))

        X_r, U_r, t_r, _ = refine_mesh(X, U, t_nodes, sho_dynamics; tol=1e-3)

        # Recompute exact values on refined mesh
        X_r_exact = [SVector(cos(tn), -sin(tn)) for tn in t_r]
        U_r_exact = [SVector(0.0) for _ in eachindex(t_r)]

        max_err_after = maximum(mesh_error(X_r_exact, U_r_exact, t_r, sho_dynamics))
        @test max_err_after < max_err_before
    end

end
