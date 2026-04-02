# # CR3BP: Lyapunov Orbit Seeding
#
# In the Circular Restricted Three-Body Problem (CR3BP), **Lyapunov orbits** are planar
# periodic orbits that exist around the collinear libration points (L₁, L₂, L₃). Finding
# them numerically requires two ingredients: a good initial guess and a differential
# corrector to refine it.
#
# This tutorial walks through both steps for the Earth–Moon L₁ point:
#
# 1. Compute the **libration point** and its local eigen-structure,
# 2. Build a **linearized seed** from the center-manifold direction,
# 3. Refine the seed into a periodic orbit with a **single-shooting corrector**.
#
# The corrected orbit will serve as the starting point for the family continuation
# in the next tutorial.
#
# *This file is compatible with **Literate.jl**: run it as a plain Julia script,
# or convert it to Markdown / Jupyter with `Literate.markdown` or `Literate.notebook`.*

using LinearAlgebra
using Serialization
using OrdinaryDiffEqVerner
using StaticArrays
using Plots

using Motion
using Motion.Continuation

# ## System parameters
#
# The CR3BP is fully characterized by the mass ratio `μ = m₂/(m₁ + m₂)`.
# Here we use the Earth–Moon value:
const μ = 1.215058560962404e-2

# ## Libration point and eigen-structure
#
# The collinear libration points are equilibria of the CR3BP equations of motion.
# We compute L₁ and evaluate the Jacobian of the vector field there:
xLP = Motion.libration_point(μ, Val(:L1))
JLP = Motion.CR3BP.jacobian(xLP, μ);

# The eigenvalues of this 6×6 Jacobian come in three pairs: one real pair
# (saddle — stable/unstable manifolds), one purely imaginary pair (in-plane center
# — Lyapunov family), and one purely imaginary pair (out-of-plane center — vertical family).
# The eigenvectors associated with the imaginary eigenvalues span the **center manifold**,
# which is where periodic orbit families originate.
L, W = eigen(JLP)

# ## Linearized Lyapunov seed
#
# A small displacement along a center eigenvector (one with a purely imaginary
# eigenvalue) produces an approximate periodic orbit — the *linearized seed*.
# We take a unit-normalized eigenvector and scale it by a small amplitude `ε`:
x0g = xLP + 1e-3 * real(W[:, 2] / norm(W[:, 2]));

# Planar Lyapunov orbits are symmetric about the x-axis. A trajectory that
# **crosses the x-axis perpendicularly** (y = 0, vₓ = 0) will, by symmetry, cross it
# perpendicularly again after half a period. We therefore keep only `x` and `vy`
# from the eigenvector displacement, zeroing out the rest:
x0 = [x0g[1], 0, 0, 0, x0g[5], 0]

# The imaginary part of the corresponding eigenvalue gives the linear oscillation
# frequency, so `2π / |Im(λ)|` is our first guess for the period:
T0 = 2π / abs(L[2])

# ## Single-shooting correction
#
# The linearized seed is only approximate — it satisfies the *linear* equations of
# motion, not the full nonlinear CR3BP. A **single-shooting corrector** iteratively
# adjusts the initial conditions (and possibly the period) until the trajectory
# closes on itself to machine precision.
#
# Because we only vary a subset of the full state, we define a **reduced layout**:
# out of the 6 state components `[x, y, z, vₓ, vy, vz]`, only `x` (index 1) and
# `vy` (index 5) are free; the period `T` is also free. This gives a reduced
# state vector `z = [x, vy, T]` of dimension 3.
layout = SingleShootingReducedLayout(6, [1, 5], true);

# ### Choosing a constraint
#
# A planar Lyapunov orbit's symmetry lets us enforce periodicity with only
# **two scalar constraints** (matching the two free state components). Two
# equivalent formulations exist:
#
# **Half-period symmetry** — require the trajectory to hit the x-axis
# perpendicularly at `T/2`:
# - `y(T/2) = 0`
# - `vₓ(T/2) = 0`
#
# **Full-period periodicity** — require the trajectory to return to its
# initial state after one full period:
# - `x(T) - x(0) = 0`
# - `vy(T) - vy(0) = 0`
#
# Here we use the full-period constraint via `Continuation.Periodicity`:
f(x, T, λ) = Motion.CR3BP.flow(μ, x, 0.0, T, Vern9(); abstol =  reltol=1e-14 );
sr = SingleShootingResidual(
	SingleShooting(f, layout), Continuation.Periodicity(layout),
);

# ### Phase constraint
#
# With 3 unknowns (`x`, `vy`, `T`) and 2 periodicity constraints the system is
# under-determined — one degree of freedom remains, corresponding to the **family
# parameter** (amplitude). To pin a unique solution we add a **natural-parameter
# constraint** that fixes `x` to its initial-guess value:
r = NaturalParameterShootingResidual(sr, 1);

# ### Solve the corrector
#
# Pack the initial guess into the reduced vector `z₀ = [x, vy, T]` and
# solve the resulting nonlinear system with Newton–Raphson:
z0 = [x0[1], x0[5], T0]
corr = Continuation.SciMLCorrector(SimpleNewtonRaphson(); abstol =  reltol=1e-10 , verbose = false);
zsol, stat = solve(r, corr, z0, z0[1]);

# ## Plot the initial guess vs the corrected orbit

# Integrate the linearized seed over one period:
sol_guess = Motion.CR3BP.build_solution(μ, x0, 0.0, T0, Vern9(); abstol =  reltol=1e-14 );
X_guess = reduce(hcat, sol_guess.(LinRange(0, T0, 1000)));

# Integrate the corrected orbit over one period:
xn, Tn = Continuation.unpack(layout, zsol)
sol = Motion.CR3BP.build_solution(μ, xn, 0.0, Tn, Vern9(); abstol =  reltol=1e-14 )
X = reduce(hcat, sol.(LinRange(0, Tn, 1000)));

# The corrected orbit (black) closes on itself, while the linearized seed (red,
# dotted) visibly drifts — demonstrating why the differential correction step
# is essential.
begin
	p = plot(framestyle = :box, xlabel = "x (-)", ylabel = "y (-)", aspect_ratio = 1, legend = :bottomright)
	scatter!(p, [xLP[1]], [xLP[2]], label = false, marker = :d, color = :red)
	plot!(p, X_guess[1, :], X_guess[2, :], color = :red, style = :dot, label = "Initial guess")
	plot!(p, X[1, :], X[2, :], color = :black, label = "Corrected orbit")
end

## hideall
cache_path = joinpath(@__DIR__, "cache", "010_L1_Lyap_seed.jls")
	mkpath(dirname(cache_path))
	serialize(cache_path,
		(; μ = μ, x = xn, T = Tn )
	)
