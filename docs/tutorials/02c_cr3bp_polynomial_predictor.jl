# # CR3BP: Polynomial continuation for planar Lyapunov families
#
# This tutorial demonstrates the new `PolynomialPredictor` on the same planar
# Lyapunov-family problem used in the natural-parameter and pseudo-arclength examples.
#
# We will:
#
# 1. build a corrected planar Lyapunov seed near **L₁**,
# 2. warm-start the continuation history with small natural-parameter steps,
# 3. switch to an adaptive **Newton divided-difference polynomial predictor**,
# 4. plot both the orbit family and the predictor's step/order history.
#
# The file is compatible with **Literate.jl**: it can be executed directly, or converted # to Markdown/Notebook as part of the docs build.
using LinearAlgebra
using SimpleNonlinearSolve
using OrdinaryDiffEqVerner
using StaticArrays
using Plots

using Motion
using Motion.Continuation

# ## Seed setup
const μ = 0.012150584269940356

LPs = Motion.libration_points(μ)
xLP = LPs[1]                      # L₁
JLP = Motion.CR3BP.jacobian(xLP, μ)
L, W = eigen(JLP);

# Linearized center-direction guess near L₁
x0g = xLP + 0.02 * real(W[:, 5] / norm(W[:, 5]));

# Planar initial condition: y=z=vx=vz=0, vy free
x0 = [x0g[1], 0, 0, 0, x0g[5], 0]
T0 = 2π / abs(L[5])

# ## Reduced variables and shooting residual
#
# We continue the reduced vector
#
#   z = [ x[1], vᵧ(0), T/2 ]
#
# and rebuild the full state `[x0; T]` via `ReducedLayout`.
layout = ReducedLayout(
	SingleShootingLayout(6),
	VarMap(7, [1, 5, 7]),
	vcat(x0, T0),
);

flow = (x, T, λ) -> Motion.CR3BP.flow(
	μ, x, 0.0, T, Vern9(); abstol = 1e-12, reltol = 1e-12,
);

sys = SingleShootingResidual(
	ShootingArc(flow, layout),
	HalfPeriodSymmetry((2, 4)),
);

zinit = [x0[1], x0[5], T0 / 2]

# ## Corrector and warm-start continuation problems
#
# The polynomial predictor uses the same pseudo-arclength correction equation as PALC,
# so we work with a square residual in `z` before adding the arclength condition.
corr = Corrector(
	SimpleNewtonRaphson(); abstol = 1e-10, reltol = 1e-10, verbose = false,
);

nat = ContinuationProblem(
	sys;
	predictor = SimpleNaturalParameter(1),
	corrector = corr,
);

poly_sys = Continuation.SimpleNaturalParameterShootingResidual(sys, 1)
predictor = PolynomialPredictor(
	ds0 = 5e-4,
	hmin = 1e-4,
	hmax = 1,
	hfail = 1e-4,
	dhmax = 1.5,
	max_degree = 4,
	err_abs = 1e-10,
	err_rel = 1e-8,
)

poly = ContinuationProblem(
	poly_sys;
	predictor = predictor,
	corrector = corr,
);

# ## Initialization of history
#
# As in the PALC tutorial, we first correct the initial seed and then take a few
# tiny natural-parameter steps to build enough history for the polynomial predictor
# to raise its interpolation order cleanly.
history = ContinuationPoint{Float64}[
	ContinuationPoint{Float64}(zinit, zinit[1]),
];

push!(history, Continuation.step!(nat, history; ds = 0.0)[1])
popfirst!(history)

for _ in 1:3
	push!(history, Continuation.step!(nat, history; ds = 1e-4)[1])
end

# ## Run the continuation with the polynomial predictor
#
# Here we let `PolynomialPredictor` adapt both the next step length and the interpolation
# degree internally. Calling `step!(poly, history)` without `ds` reuses the predictor's
# current `stepsize(predictor)`.
nsteps = 120
step_hist = Float64[stepsize(predictor)]
deg_hist = Int[polynomial_degree(predictor)]

for _ in 1:nsteps
	point, stat = Continuation.step!(poly, history)
	stat.success || error("Polynomial continuation step failed")
	push!(history, point)
	push!(step_hist, stepsize(predictor))
	push!(deg_hist, polynomial_degree(predictor))
end

# ## Plot the orbit family
orbit_plot = plot(
	framestyle = :box,
	xlabel = "x (-)",
	ylabel = "y (-)",
	aspect_ratio = 1,
	legend = :bottomright,
	dpi = 200,
)

scatter!(orbit_plot, [xLP[1]], [xLP[2]], label = false, marker = :d, color = :red)

for i in 1:10:length(history)
	point = history[i]
	xn = [point.z[1], 0, 0, 0, point.z[2], 0]
	Tn = 2 * point.z[3]

	sol = Motion.CR3BP.build_solution(
		μ, xn, 0.0, Tn, Vern9(); abstol = 1e-12, reltol = 1e-12,
	)
	X = reduce(hcat, sol.(LinRange(0, Tn, 400)))

	plot!(orbit_plot, X[1, :], X[2, :], color = :black, linewidth = 0.6, label = false)
end

orbit_plot

# ## Plot the predictor diagnostics
#
# The predictor starts from degree 1 and may increase or decrease the interpolation
# order depending on how well the latest corrected point matches lower-order fits.
step_plot = plot(
	1:length(step_hist),
	step_hist,
	framestyle = :box,
	label = false,
	linewidth = 2,
	xlabel = "Continuation step",
	ylabel = "Step size",
	title = "Adaptive step size",
	dpi = 200,
)

##

degree_plot = plot(
	1:length(deg_hist),
	deg_hist,
	framestyle = :box,
	label = false,
	linewidth = 2,
	color = :red,
	xlabel = "Continuation step",
	ylabel = "Degree",
	title = "Polynomial order",
	dpi = 200,
)
