# # CR3BP: Natural-Parameter Continuation for Lyapunov Families
#
# Periodic orbits in the CR3BP come in **one-parameter families**: once a single
# member is known (e.g. from the seeding tutorial), neighbouring members can be
# found by slowly varying a parameter and correcting at each step. This process
# is called **numerical continuation**.
#
# The simplest variant is **natural-parameter continuation**, where we march along
# a physical quantity — here the initial x-position `x(0)` — taking small steps
# and re-solving the shooting problem at each one.
#
# This tutorial demonstrates how to:
#
# 1. Load the seed orbit computed in the previous tutorial,
# 2. Set up a **single-shooting residual** with a **half-period symmetry** constraint,
# 3. Run a natural-parameter continuation loop to trace a family of planar
#    Lyapunov orbits around Earth–Moon L₁,
# 4. Plot the resulting orbit family.
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

# ## Load seed data
#
# We deserialize the corrected orbit from the first tutorial. This gives us a
# validated initial condition `x₀` and period `T₀` to start the continuation from.
const seed_data = deserialize(joinpath(@__DIR__, "cache", "010_L1_Lyap_seed.jls"))

const μ = seed_data.μ
x0 = seed_data.x
T0 = seed_data.T

# ## Continuation setup
#
# ### Reduced layout
#
# As in the seeding tutorial, we work with a reduced state vector `z = [x, vy, T]`
# containing only the free components. The full 6D state is reconstructed by the
# layout when needed (all fixed components default to zero, which is consistent
# with the planar Lyapunov symmetry).
layout = SingleShootingReducedLayout(6, [1, 5], true);

# ### Shooting residual with half-period symmetry
#
# This time we use the **half-period symmetry** constraint instead of the
# full-period periodicity used in the seed tutorial. Because planar Lyapunov
# orbits are symmetric about the x-axis, it suffices to integrate for only
# *half* a period and require:
#
# - `y(T/2) = 0`   — the trajectory returns to the x-axis,
# - `vₓ(T/2) = 0`  — it crosses perpendicularly.
#
# This halves the integration time and often improves convergence.
f(x, T, λ) = Motion.CR3BP.flow(μ, x, 0.0, T, Vern9(); abstol =  reltol=1e-14 );
sr = SingleShootingResidual(
	SingleShooting(f, layout), Continuation.HalfPeriodSymmetry([2, 4]),
);

# ### Initial reduced state
#
# Since we are using the half-period constraint, the reduced state stores the
# *half*-period rather than the full period:
# - `z[1] = x(0)`
# - `z[2] = vy(0)`
# - `z[3] = T/2`
z0 = [x0[1], x0[5], T0/2]

# ### Continuation history
#
# The continuation algorithm maintains a **history stack** of previously converged
# points. Each `ContinuationPoint` bundles the reduced state `z` with the current
# value of the continuation parameter `λ` (here `λ = x(0)`). The predictor uses
# this history to extrapolate an initial guess for the next step.
history = ContinuationPoint{Float64}[ContinuationPoint{Float64}(z0, z0[1]),]

# ### Continuation problem
#
# A `ContinuationProblem` ties together three components:
#
# - **System** (`sr`): the shooting residual that encodes the dynamics and constraints,
# - **Predictor** (`SimpleNaturalParameter`): generates an initial guess for the next
#   point by incrementing `z[1]` (the x-position) by a fixed step `ds`,
# - **Corrector**: a Newton-based solver that refines the prediction
#   until the residual is below tolerance.
prob = ContinuationProblem(
	sr;
	predictor = SimpleNaturalParameter(1, -1),
	corrector = Continuation.SciMLCorrector(; abstol =  reltol=1e-10 ),
)

# ## Run the continuation
#
# We march in the direction of *decreasing* `x(0)` (note `sign = -1` in the
# predictor above), taking 500 steps of size `Δs`. At each step,
# `Continuation.step!` predicts, corrects, and returns the new converged point.
Δs = 2e-4
nsteps = 500

for i ∈ 1:nsteps
	push!(history, Continuation.step!(prob, history; ds = Δs)[1])
end

# ## Plot the orbit family
begin
	p = plot(
		framestyle = :box,
		xlabel = "x (-)", ylabel = "y (-)",
		aspect_ratio = 1,
		legend = :bottomright,
		dpi = 200,
	)

	for i in 1:50:length(history)
		point = history[i]
		xn, Tn = Continuation.unpack(layout, point.z)

		sol = Motion.CR3BP.build_solution(
			μ, xn, 0.0, 2Tn, Vern9(); abstol =  reltol = 1e-14 ,
		)
		X = reduce(hcat, sol.(LinRange(0, 2Tn, 1000)))

		plot!(p, X[1, :], X[2, :], color = :black, linewidth = 0.5, label = false)
	end
	p
end
