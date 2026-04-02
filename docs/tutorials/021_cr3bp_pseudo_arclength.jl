# # CR3BP: Pseudo-Arclength Continuation for Lyapunov Families
#
# Natural-parameter continuation (the previous tutorial) is simple but has a
# fundamental limitation: it fails at **turning points** (folds) where the
# family curve doubles back on itself in the continuation parameter. At a fold,
# the Jacobian with respect to the natural parameter becomes singular and the
# corrector diverges.
#
# **Pseudo-arclength continuation (PALC)** overcomes this by parameterizing the
# family curve by its *arclength* `s` rather than a single state component.
# The predictor steps along the **tangent** to the solution curve, and an
# additional **arclength constraint** replaces the natural-parameter constraint.
# This keeps the Jacobian non-singular even at folds, allowing the continuation
# to smoothly trace through turning points.
#
# This tutorial demonstrates how to:
#
# 1. Set up the same shooting residual used in the natural-parameter tutorial,
# 2. Replace the predictor and constraint with a pseudo-arclength formulation,
# 3. Trace the L₁ Lyapunov family with larger steps and fewer iterations.
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
# We use the same **half-period symmetry** constraint as in the natural-parameter
# tutorial. Because planar Lyapunov orbits are symmetric about the x-axis, it
# suffices to integrate for only *half* a period and require:
#
# - `y(T/2) = 0`   — the trajectory returns to the x-axis,
# - `vₓ(T/2) = 0`  — it crosses perpendicularly.
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
# value of the continuation parameter `λ`. The predictor uses this history to
# compute the tangent direction for the next step.
history = ContinuationPoint{Float64}[ContinuationPoint{Float64}(z0, z0[1]),]

# ### Continuation problem
#
# The key difference from the natural-parameter tutorial is the **predictor**.
# `SimplePseudoArcLength` computes the tangent to the solution curve by finding
# the null space of the shooting Jacobian at the current point. The prediction
# step is then `z_pred = z + ds * t`, where `t` is the unit tangent vector.
#
# During correction, a **pseudo-arclength constraint** `dot(t, z - z₀) = ds`
# is appended to the shooting residual, replacing the natural-parameter
# constraint. This constrains the corrector to a hyperplane perpendicular to
# the tangent, ensuring it converges to the *nearest* family member rather than
# sliding along the curve.
prob = ContinuationProblem(
	sr;
	predictor = SimplePseudoArcLength(sr),
	corrector = Continuation.SciMLCorrector(; abstol =  reltol=1e-10 ),
)

# ## Run the continuation
#
# Because PALC follows the curve tangent, it can take much **larger steps** than
# natural-parameter continuation without losing convergence. Here we use
# `Δs = 5e-2` (compare with `2e-4` in the previous tutorial) and need only
# 50 steps to cover a comparable portion of the family.
Δs = 5e-2
nsteps = 50

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

	for i in 1:5:length(history)
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
