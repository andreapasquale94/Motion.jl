# # CR3BP: Pseudo-arclength continuation for L1-L2 Lyapunov orbits
#
# This tutorial demonstrates **pseudo-arclength continuation (PALC)** for Lyapunov orbits in the CR3BP 
# using the `Motion.Continuation` module.
#
# The file is compatible with **Literate.jl**: it can be executed as-is, or converted
# into markdown/notebook.

using LinearAlgebra
using Serialization
using SimpleNonlinearSolve
using OrdinaryDiffEqVerner
using StaticArrays
using Plots

using Motion
using Motion.Continuation

# ## Seed setup
const μ = 0.012150584269940356

LPs = Motion.libration_points(μ)

# ## Reduced decision variables setup
#
# For planar Lyapunov orbits we can exploit symmetry and resolve the problem with reduced variables set:
#
#   z = [ x[1], vᵧ(0), T/2 ]
layout = ReducedLayout(
	SingleShootingLayout(6), VarMap(7, [1, 5, 7]), zeros(7)
);

# Flow map: propagate from t=0 to t=T (or T/2, depending on the arc object)
flow = (x, T, λ) -> Motion.CR3BP.flow(
	μ, x, 0.0, T, Vern9(); abstol =  reltol = 1e-12,
);

# Next, define shooting residula model:
sys = SingleShootingResidual(
	ShootingArc(flow, layout), HalfPeriodSymmetry((2, 4)),
);

# ## Corrector setup
#
# Continuation steps use a predictor (here natural parameter / PALC) and a corrector
# (Newton) to land back on the solution manifold.
corr = Corrector(
	SimpleNewtonRaphson(); abstol =  reltol = 1e-9
);

# ## Continuation problems
#
# PALC needs two consecutive points and a *square* residual in z. We add the
# natural-parameter equation to the shooting residual so that, when PALC adds its
# arclength equation, the system stays square in [z; λ].
#
# - `nat` is used only to generate the first two corrected points.
# - `palc` is used for the rest of the continuation.

nat = ContinuationProblem(
	sys;
	predictor = SimpleNaturalParameter(1),
	corrector = corr,
);

palc_sys = Continuation.SimpleNaturalParameterShootingResidual(sys, 1)
palc = ContinuationProblem(
	palc_sys;
	predictor = PseudoArcLength(),
	corrector = corr,
);

## Run of the continuation 

history = Vector{ContinuationPoint{Float64}}[]
nsteps = 300

for idx in [1, 2]
	# Create seed
	@show xLP = LPs[idx]
	JLP = Motion.CR3BP.jacobian(xLP, μ)
	L, W = eigen(JLP)
    if idx == 1 
	    x0g = xLP + 0.01 * real(W[:, 5] / norm(W[:, 5]))
    else 
	    x0g = xLP + 0.01 * real(W[:, 3] / norm(W[:, 3]))
    end
	@show x0 = [x0g[1], 0, 0, 0, x0g[5], 0]
	T0 = 2π / abs(L[5])
	zinit = [x0[1], x0[5], T0/2]

    # Initialize continuation history
	history_i = ContinuationPoint{Float64}[ContinuationPoint(zinit, zinit[1])]

	# First: correct the initial point tightly.
	push!(history_i, Continuation.step!(nat, history_i; ds = 0)[1])
	popfirst!(history_i);
	# Second: take one small natural-parameter step to seed the PALC tangent.
	Δs = idx == 1 ? 5e-3 : 3e-3
	push!(history_i, Continuation.step!(nat, history_i; ds = 1e-4)[1]);

	# Run the continuation
	for i ∈ 1:nsteps
		push!(history_i, Continuation.step!(palc, history_i; ds = Δs)[1])
	end

	push!(history, history_i)
end;

# Store the results for later tutorials
begin
	cache_path = joinpath(@__DIR__, "cache", "031_L1_lyap_palc.jls")
	mkpath(dirname(cache_path))
	serialize(cache_path, history[1])
end
begin
	cache_path = joinpath(@__DIR__, "cache", "031_L2_lyap_palc.jls")
	mkpath(dirname(cache_path))
	serialize(cache_path, history[2])
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

    for k in [1, 2]
	    scatter!(p, [LPs[k][1]], [LPs[k][2]], label = false, marker = :d, color = :red)

        for i in 1:20:nsteps
            point = history[k][i]
            xn = [point.z[1], 0, 0, 0, point.z[2], 0]
            Tn = 2 * point.z[3]

            sol = Motion.CR3BP.build_solution(
                μ, xn, 0.0, Tn, Vern9(); abstol =  reltol = 1e-10,
            )
            X = reduce(hcat, sol.(LinRange(0, Tn, 1000)))

            plot!(p, X[1, :], X[2, :], color = :black, linewidth = 0.5, label = false)
        end
    end
end
p