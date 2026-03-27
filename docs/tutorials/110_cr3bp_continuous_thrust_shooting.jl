# # CR3BP: Continuous-thrust shooting from the impulsive solution
#
# This tutorial is the simplest continuous-thrust transcription of tutorial 03:
#
# 1. load the converged impulsive multiple-shooting solution,
# 2. convert each impulse `Δv_k` into a constant segment acceleration `u_k = Δv_k / Δt_k`,
# 3. keep the shooting time grid fixed,
# 4. solve a bounded continuous-thrust NLP with the path constraint
#
# \[
# \|u_k\| \le u_{\max}, \qquad k = 1,\ldots,N-1.
# \]
#
# The thrust cap is chosen from a simple physical reference:
# a **1 N chemical thruster** on a **1000 kg spacecraft**, converted into
# nondimensional Earth-Moon CR3BP acceleration units.

using Motion
using LinearAlgebra
using OrdinaryDiffEqVerner
using Serialization

using Optimization, OptimizationIpopt
using ForwardDiff

function earth_moon_accel_unit_mps2()
	du_m = 384_400.0e3
	t_sid_s = 27.321661 * 86_400.0
	tu_s = t_sid_s / (2π)
	return du_m / tu_s^2
end

function project_controls!(vars, iU, umax)
	for k in 1:(size(iU, 2) - 1)
		idx = iU[:, k]
		uk = vars[idx]
		nuk = norm(uk)
		if nuk > umax
			vars[idx] .= uk .* (umax / nuk)
		end
	end
	vars[iU[:, end]] .= 0.0
	return vars
end

function solve_continuous_thrust(xx0, μ, vN, vnx, vnu, xstart, xfinal, dt_ref, umax)
	it0, idt, iX, iU = Motion.MultipleShooting.indexes(vN, vnx, vnu)
	nx = size(iX, 1)
	nseg = length(idt)
	nvars = iU[end]

	flow = (x, u, t0, t1) -> Motion.CR3BP.flow_const_thrust(
		μ, x, u, t0, t1, Vern9(); reltol = 1e-8, abstol = 1e-8,
	)

	objective = (x, p) -> Motion.MultipleShooting.objective(
		x, vN, vnx, vnu, Val(:PIECEWISE_CONST_FUEL),
	)

	constraints = (out, x, p) -> begin
		defects = Motion.MultipleShooting.defects(x, flow, vN, vnx, vnu, Val(:Forward))
		control_normsq = Motion.MultipleShooting.control_normsq(x, vN, vnx, vnu)
		out[1:(nx * nseg)] .= defects
		out[(nx * nseg + 1):end] .= control_normsq
		nothing
	end

	lb = fill(-Inf, nvars)
	ub = fill(Inf, nvars)
	lb[it0] = 0.0
	ub[it0] = 0.0
	lb[idt] .= dt_ref
	ub[idt] .= dt_ref
	lb[iX[:, 1]] .= xstart
	ub[iX[:, 1]] .= xstart
	lb[iX[:, end]] .= xfinal
	ub[iX[:, end]] .= xfinal
	lb[iU[:, end]] .= 0.0
	ub[iU[:, end]] .= 0.0

	clb = vcat(fill(0.0, nx * nseg), fill(-Inf, nseg))
	cub = vcat(fill(0.0, nx * nseg), fill(umax^2, nseg))

	optfun = OptimizationFunction(objective, AutoForwardDiff(); cons = constraints)
	prob = OptimizationProblem(optfun, xx0; lb = lb, ub = ub, lcons = clb, ucons = cub)

	return solve(
		prob,
		IpoptOptimizer(
			acceptable_tol = 1e-6,
			constr_viol_tol = 1e-9,
			mu_strategy = "adaptive",
			hessian_approximation = "limited-memory",
			limited_memory_max_history = 5,
			additional_options = Dict("print_level" => 5),
		);
		maxiters = 1500,
		verbose = 5
	)
end

# ## Load the cached impulsive solution from tutorial 03
cache_path = joinpath(@__DIR__, "cache", "03_impulsive_transfer_seed.jls")
isfile(cache_path) || error(
	"Impulsive seed not found. Run tutorial 03 first so the cache is written to $(cache_path)",
)

seed = deserialize(cache_path)
μ = seed.μ
vN = Val(seed.N)
vnx = Val(seed.nx)
vnu = Val(seed.nu)

xx_imp = seed.decision
_, dt_imp, _, U_imp, _ = Motion.MultipleShooting.variables(xx_imp, vN, vnx, vnu)

# ## Build the continuous-thrust initial guess
#
# Each segment keeps the same duration as in tutorial 03.
A0 = zeros(eltype(xx_imp), seed.nu, seed.N)
for k in 1:(seed.N - 1)
	A0[:, k] .= U_imp[:, k] ./ max(dt_imp[k], 1e-6)
end

_, _, X_imp, _, _ = Motion.MultipleShooting.variables(xx_imp, vN, vnx, vnu)
xx_ct = vcat(
	0.0,
	collect(dt_imp),
	reshape(Matrix(X_imp), seed.nx * seed.N),
	reshape(A0, seed.nu * seed.N),
)

_, _, _, iU = Motion.MultipleShooting.indexes(vN, vnx, vnu)

# ## Convert a physical thrust cap into nondimensional CR3BP units
spacecraft_mass_kg = 1000.0
thrust_max_N = 1.0
accel_unit_mps2 = earth_moon_accel_unit_mps2()
umax = (thrust_max_N / spacecraft_mass_kg) / accel_unit_mps2

project_controls!(xx_ct, iU, umax)

seed_max_thrust_N = maximum(norm.(eachcol(A0[:, 1:(end - 1)]))) * accel_unit_mps2 * spacecraft_mass_kg
@info "Continuous-thrust setup" thrust_max_N spacecraft_mass_kg umax seed_max_thrust_N

# ## Solve the continuous-thrust transcription
sol = solve_continuous_thrust(
	xx_ct, μ, vN, vnx, vnu,
	seed.xstart, seed.xfinal, dt_imp, umax,
)

# ## Summarize the result
flow = (x, u, t0, t1) -> Motion.CR3BP.flow_const_thrust(
	μ, x, u, t0, t1, Vern9(); reltol = 1e-8, abstol = 1e-8,
)
defects = Motion.MultipleShooting.defects(sol.u, flow, vN, vnx, vnu, Val(:Forward))
control_normsq = Motion.MultipleShooting.control_normsq(sol.u, vN, vnx, vnu)
control_norm = sqrt.(control_normsq)
thrust_norm_N = control_norm .* accel_unit_mps2 .* spacecraft_mass_kg
fuel_like_cost = Motion.MultipleShooting.objective(sol.u, vN, vnx, vnu, Val(:PIECEWISE_CONST_FUEL))

@info "Continuous-thrust result" retcode = sol.retcode fuel_like_cost max_defect = maximum(abs.(defects)) max_thrust_N = maximum(thrust_norm_N)
