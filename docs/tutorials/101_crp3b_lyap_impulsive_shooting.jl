using Motion
using LinearAlgebra
using OrdinaryDiffEqVerner
using Plots
using Serialization
using StaticArrays

using Optimization, OptimizationIpopt
using ForwardDiff

# ## Problem setup
#
# We work in nondimensional CR3BP units with mass parameter μ.
#

const opt_seed = deserialize(joinpath(@__DIR__, "cache", "100_impulsive_transfer_seed.jls"))
const μ = opt_seed.μ

# A small plot helper: primaries + L1/L2.
function plot_xy(μ)
	lps = Motion.libration_points(μ)[1:2]
	p = plot(framestyle = :box, xlabel = "x (-)", ylabel = "y (-)", aspect_ratio = 1)
	scatter!([1-μ], [0], label = false, marker = :o, color = :grey)
	for i in eachindex(lps)
		scatter!(p, [lps[i][1]], [lps[i][2]], label = false, marker = :d, color = :red)
	end
	return p
end

## Select and propagate departure/arrival orbits 

const L1_fam = deserialize(joinpath(@__DIR__, "cache", "031_L1_lyap_palc.jls")).data
const L2_fam = deserialize(joinpath(@__DIR__, "cache", "031_L2_lyap_palc.jls")).data

x0_dep, T_dep = L1_fam[15]
x0_arr, T_arr = L2_fam[15]

orb_dep = Motion.CR3BP.build_solution(μ, x0_dep, 0.0, T_dep, Vern9(); reltol = 1e-12, abstol = 1e-12);
orb_arr = Motion.CR3BP.build_solution(μ, x0_arr, 0.0, T_arr, Vern9(); reltol = 1e-12, abstol = 1e-12);

## Decision vector layout
nx = opt_seed.nx 
nu = opt_seed.nu 
N = opt_seed.N
x0_seed = opt_seed.decision

vN  = Val(N)
vnx = Val(nx)
vnu = Val(nu);

it0, idt, iX, iU = Motion.MultipleShooting.indexes(vN, vnx, vnu)

# ## Segment flow with impulses
B = zeros(nx, nu)
B[(nx-nu+1):end, :] .= I(nu)  # add to velocity components only

flow = (x, u, t0, t1) -> begin
	x0 = x + B*u
	Motion.CR3BP.flow(μ, x0, t0, t1, Vern9(); reltol = 1e-8, abstol = 1e-8)
end;

# ## Define objective and constraints 

objective = (x, p) -> Motion.MultipleShooting.objective(@view(x[3:end]), vN, vnx, vnu, Val(:FUEL))

function wrap_time(t, T)
    rem, dt = divrem(t, 1)
    if iszero(rem)
        return t*T 
    else 
        return dt*T 
    end
end

constraints = (out, x, p) -> begin
    t_dep = wrap_time(x[1], T_dep)
    t_arr = wrap_time(x[2], T_arr)

    x_dep = orb_dep(t_dep)
    x_arr = orb_arr(t_arr)

    z = @view(x[3:end])
    X = z[iX]

    out[1:6] .= (x_dep - X[:, 1])
    out[7:12] .= (x_arr - X[:, end])
	out[13:end] .= Motion.MultipleShooting.defects(z, flow, vN, vnx, vnu, Val(:Forward))
	nothing
end;

# Create bounds for lb/ub and constraints
nvars = length(x0_seed) + 2
lb = fill(-Inf, nvars)
ub = fill(Inf, nvars)

# dep/arr bounds 
lb[1:2] .= -2 
ub[1:2] .= 2

# dt bounds
lb[idt] .= 1e-12
ub[idt] .= 5.0;

# equality defects
clb = zeros(nx*(N-1) + 2*6)
cub = zeros(nx*(N-1) + 2*6)

# Initial guess 
x0 = vcat(-0.25, 0, x0_seed)

# ## Create and solve the NLP

optfunc = OptimizationFunction(objective, AutoForwardDiff(); cons = constraints);

prob = OptimizationProblem(optfunc, x0;
	lb = lb, ub = ub,
	lcons = clb, ucons = cub,
);

sol = solve(prob,
	IpoptOptimizer(
		acceptable_tol = 1e-6,
		mu_strategy = "adaptive",
		hessian_approximation = "limited-memory",
		limited_memory_max_history = 5,
        constr_viol_tol = 1e-10
	);
	maxiters = 5000,
	verbose = 5
);

## Plot 

p = plot_xy(μ)

Xdep = reduce(hcat, orb_dep.(LinRange(0, T_dep, 400)))
Xarr = reduce(hcat, orb_arr.(LinRange(0, T_arr, 400)))

plot!(p, Xdep[1, :], Xdep[2, :], color = :black, linewidth = 0.6, label = false)
plot!(p, Xarr[1, :], Xarr[2, :], color = :black, linewidth = 0.6, label = false)

_, _, Xo, Uo, to = Motion.MultipleShooting.variables(sol.u[3:end], vN, vnx, vnu)
scatter!(p, Xo[1, :], Xo[2, :], ms = 2, label = false, color = :green)

for k in 1:(N-1)
	xk = @view Xo[:, k]
	uk = @view Uo[:, k]

	x0seg = xk + B*uk
	solseg = Motion.CR3BP.build_solution(μ, x0seg, to[k], to[k+1], Vern9(); reltol = 1e-8, abstol = 1e-8)

	τ = range(to[k], to[k+1], length = 100)
	Xseg = reduce(hcat, solseg.(τ))

	plot!(p, Xseg[1, :], Xseg[2, :], label = false, linewidth = 0.75, color = :grey, style = :dash)
end
p