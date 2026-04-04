# # CR3BP: Halo Orbit Family Continuation
#
# With a single corrected halo orbit in hand (from the previous tutorial), we can
# now trace the entire **southern halo family** using pseudo-arclength continuation.
# Halo orbits are three-dimensional, so the reduced state vector has one more
# component than the planar Lyapunov case: `z = [x, z, vy, T/2]`.
#
# This tutorial demonstrates how to:
#
# 1. Load the halo seed and set up a 3D shooting residual,
# 2. Run pseudo-arclength continuation to trace the southern halo family,
# 3. Plot the resulting family in the x–z plane.

using LinearAlgebra
using Serialization
using OrdinaryDiffEqVerner
using StaticArrays
using Plots

using Motion
using Motion.Continuation

# ## Load seed data
#
# We deserialize the corrected halo orbit from the previous tutorial.
const seed_data = deserialize(joinpath(@__DIR__, "cache", "031_L1_SHalo_seed.jls"))

const μ = seed_data.μ
x0 = seed_data.x
T0 = seed_data.T

# ## Continuation setup
#
# ### Reduced layout
#
# Halo orbits have three free state components: `x` (index 1), `z` (index 3),
# and `vy` (index 5). Together with the period this gives a reduced vector
# `z = [x, z, vy, T/2]` of dimension 4.
layout = SingleShootingReducedLayout(6, [1, 3, 5], true);

# ### Half-period symmetry constraint
#
# Halo orbits are symmetric about the xz-plane. At the half-period the trajectory
# must cross the xz-plane perpendicularly, which imposes three conditions:
#
# - `y(T/2) = 0`   — returns to the xz-plane,
# - `vₓ(T/2) = 0`  — perpendicular crossing (in-plane),
# - `vz(T/2) = 0`  — perpendicular crossing (out-of-plane).
f(x, T, λ) = Motion.CR3BP.flow(μ, x, 0.0, T, Vern9(); abstol=1e-14, reltol=1e-14 );
sr = SingleShootingResidual(
	SingleShooting(f, layout), Continuation.HalfPeriodSymmetry([2, 4, 6]),
);

# ### Initial reduced state and history
z0 = [x0[1], x0[3], x0[5], T0/2]

history = ContinuationPoint{Float64}[ContinuationPoint{Float64}(z0, 0.0),]

# ### Continuation problem
#
# We use `SimplePseudoArcLength` as in the Lyapunov case. The tangent predictor
# and arclength constraint work identically regardless of the orbit family's
# dimensionality — only the reduced vector size changes.
prob = ContinuationProblem(
	sr;
	predictor = SimplePseudoArcLength(sr),
	corrector = Continuation.SciMLCorrector(; abstol=1e-10, reltol=1e-10 ),
)

# ## Run the continuation
Δs = 5e-3
nsteps = 400

for i ∈ 1:nsteps
	push!(history, Continuation.step!(prob, history; ds = Δs)[1])
end

# ## Plot the halo family
#
# We plot in the x–z plane to highlight the three-dimensional character of
# halo orbits. Near the bifurcation point the orbits are nearly planar; as
# the family evolves, the out-of-plane amplitude grows significantly.
begin
	p = plot(
		framestyle = :box,
		xlabel = "x (-)", ylabel = "z (-)",
		aspect_ratio = 1,
		legend = :bottomright,
		dpi = 200,
	)

	for i in 1:10:length(history)
		point = history[i]
		xn, Tn = Continuation.unpack(layout, point.z)

		sol = Motion.CR3BP.build_solution(
			μ, xn, 0.0, 2Tn, Vern9(); abstol=1e-14, reltol=1e-14 ,
		)
		X = reduce(hcat, sol.(LinRange(0, 2Tn, 1000)))

		plot!(p, X[1, :], X[3, :], color = :black, linewidth = 0.5, label = false)
	end
	p
end

cache_path = joinpath(@__DIR__, "cache", "032_L1_SHalo.jls")
	mkpath(dirname(cache_path))
	serialize(cache_path,
		(; μ = μ, data = [Continuation.unpack(layout, p.z) for p in history] )
) # hide
