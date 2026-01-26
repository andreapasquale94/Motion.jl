# # CR3BP: Lyapunov family seeding
#
# This tutorial shows how to:
#
# 1. pick a libration point (L₁ by default),
# 2. build a **linearized center-manifold seed** (a Lyapunov-like initial guess),
# 3. refine it into a periodic orbit with a **single–shooting corrector**
#
# The code is written to be compatible with **Literate.jl**: run it as a normal Julia
# script, or convert it to Markdown/Notebook with Literate.

using Motion
using LinearAlgebra
using SimpleNonlinearSolve
using OrdinaryDiffEqVerner
using StaticArrays
using Plots

# ## Problem setup
const μ = 0.012150584269940356

# Libration points (L₁, L₂, L₃, L₄, L₅):
LPs = Motion.libration_points(μ)

# Pick L₁ as the seed location:
xLP = LPs[1];

# Jacobian of the CR3BP vector field at the libration point:
JLP = Motion.CR3BP.jacobian(xLP, μ);

# Eigen-structure (stable/unstable/center directions):
L, W = eigen(JLP)

# ## A linearized Lyapunov seed (center manifold direction)
#
# For planar Lyapunov orbits around L₁/L₂, a standard seed is a small displacement
# along one of the **center** eigenvectors (purely imaginary eigenvalues), and then
# we enforce the planar symmetry (`y=z=ẏ=ż=0` initially).
x0g = xLP + 0.1 * real(W[:, 5] / norm(W[:, 5]));

# Impose planar symmetry:
# - start on the x-axis (`y=z=0`)
# - start with `vx=ż=0`
# - leave vy free (we'll correct it)
x0 = [x0g[1], 0, 0, 0, x0g[5], 0]

# Linearized period guess from the imaginary eigenvalue:
T0 = 2π / abs(L[5])

# ## Single–shooting correction (two unknowns → two constraints)
#
# A planar Lyapunov orbit has a symmetry that lets us enforce periodicity with *two scalar constraints*, typically:
#
# - `y(T/2) = 0`
# - `vₓ(T/2) = 0`
#
# while solving for:
#
# - `vᵧ(0)` (initial transverse velocity)
# - `T` (the period)
layout = Motion.Continuation.ReducedLayout(
	Motion.Continuation.SingleShootingLayout(6),
	Motion.Continuation.VarMap(7, [5, 7]),
	vcat(x0, T0),
);

# Create a shooting segment
flow = (x, T, λ) -> Motion.CR3BP.flow(μ, x, 0.0, T/2, Vern9(); abstol = reltol=1e-12 );
shooter = Motion.Continuation.SingleShooting(flow; layout = layout)

func! = (out, z, p) -> begin
	L = shooter.layout
	u = Motion.Continuation.unpack(L, z)
	xf = Motion.Continuation.shoot(shooter, u.x0, u.T, 0.0)
	out[1] = xf[2]
	out[2] = xf[4]
	return out
end;

# Solve for `z`
prob = NonlinearProblem(func!, vcat(x0[5], T0))
sol = solve(prob, SimpleNewtonRaphson(); verbose = true, abstol = reltol=1e-10);

# We'll integrate the initial guess:
sol_guess = Motion.CR3BP.build_solution(μ, x0, 0.0, T0, Vern9(); abstol = reltol=1e-12 )
X_guess = reduce(hcat, sol_guess.(LinRange(0, T0, 1000)));

# Integrate the corrected orbit:
xn = [x0g[1], 0, 0, 0, sol.u[1], 0]
Tn = sol.u[2]
sol = Motion.CR3BP.build_solution(μ, xn, 0.0, Tn, Vern9(); abstol = reltol=1e-12 )
X = reduce(hcat, sol.(LinRange(0, Tn, 1000)));

# Plot: initial guess vs corrected periodic orbit
begin
	p = plot(framestyle = :box, xlabel = "x (-)", ylabel = "y (-)", aspect_ratio = 1, legend = :bottomright)
	scatter!(p, [xLP[1]], [xLP[2]], label = false, marker = :d, color = :red)
	plot!(p, X_guess[1, :], X_guess[2, :], color = :red, style = :dot, label = "Initial guess")
	plot!(p, X[1, :], X[2, :], color = :black, label = "Corrected orbit")
	plot!(p, xlim = (0.65, 1.0), ylim = (-0.1, 0.1))
end
