# # CR3BP: Connecting L1 → L2 Manifolds with DDP
#
# This tutorial solves the same L1→L2 manifold connection problem
# as `100_cr3bp_impulsive_shooting.jl`, but uses **constrained
# Differential Dynamic Programming** (DDP/iLQR) with an augmented-
# Lagrangian outer loop instead of Ipopt.
#
# ## Contents
# 1. CR3BP setup and manifold arc generation (same as tutorial 100)
# 2. Problem formulation as a DDP problem
# 3. Solve with `Motion.DDP.solve`
# 4. Plot the stitched trajectory
#
# ## Deps
# - `Motion`
# - `OrdinaryDiffEqVerner`
# - `Plots`

using Motion
using Motion.DDP
using LinearAlgebra
using OrdinaryDiffEqVerner
using Plots
using StaticArrays

# ## Problem setup
const μ = 0.012150584269940356

function plot_xy(μ)
	lps = Motion.libration_points(μ)[1:2]
	p = plot(framestyle = :box, xlabel = "x (-)", ylabel = "y (-)", aspect_ratio = 1)
	scatter!([1-μ], [0], label = false, marker = :o, color = :grey)
	for i in eachindex(lps)
		scatter!(p, [lps[i][1]], [lps[i][2]], label = false, marker = :d, color = :red)
	end
	return p
end

LPs = Motion.libration_points(μ)
Lp1 = LPs[1]
Lp2 = LPs[2];

# ## Approximate unstable arc from L1 (forward)
tfu = 1.2π
M1 = Motion.CR3BP.jacobian(Lp1, μ)
eig1 = eigen(M1)
w_u = real(eig1.vectors[:, end])

x0u = Lp1 .+ 1e-3 * w_u
sol_u = Motion.CR3BP.build_solution(μ, x0u, 0.0, tfu, Vern9(); abstol = 1e-12, reltol = 1e-12);

# ## Approximate stable arc into L2 (backward)
tsu = -2π
M2 = Motion.CR3BP.jacobian(Lp2, μ)
eig2 = eigen(M2)
w_s = real(eig2.vectors[:, end])

x0s = Lp2 .+ 1e-2 * w_s
sol_s = Motion.CR3BP.build_solution(μ, x0s, 0.0, tsu, Vern9(); abstol = 1e-12, reltol = 1e-12);

# ## Node placement via stretch measure
Δs = 0.1

sol_w_u = Motion.compute_stretch(sol_u, 0, tfu, Δs, Vern9())
sol_w_s = Motion.compute_stretch(sol_s, 0, tsu, -Δs, Vern9())

# ## Build initial trajectory guess
# Concatenate unstable (forward) and stable (backward, reversed) arcs.
t_nodes = vcat(sol_w_u.t[1:end-1], reverse(sol_w_s.t))
X_nodes = vcat(sol_u.(sol_w_u.t[1:end-1]), reverse(sol_s.(sol_w_s.t)))

N  = length(X_nodes)
nx = 6
nu = 3

# Convert to SVectors
X0 = [SVector{nx}(X_nodes[k]) for k in 1:N]
U0 = [SVector{nu}(1e-9 * ones(nu)) for _ in 1:(N-1)]
t  = collect(Float64, t_nodes)

# Ensure time is monotonically increasing with small positive dt
for k in 2:N
    if t[k] <= t[k-1]
        t[k] = t[k-1] + 0.5
    end
end

# ## Segment flow with impulses (same as tutorial 100)
# Control u is a velocity impulse applied before propagation.
const B_imp = let B = zeros(nx, nu); B[(nx-nu+1):end, :] .= I(nu); SMatrix{nx,nu}(B) end

flow = (x::SVector{6,T}, u::SVector{3,T}, t0, t1) where T -> begin
	x0 = x + B_imp * u
	SVector{6,T}(Motion.CR3BP.flow(μ, x0, t0, t1, Vern9(); reltol = 1e-8, abstol = 1e-8))
end

# ## DDP Problem formulation
#
# - **Stage cost**: fuel ∑ ‖uₖ‖ (smooth L1)
# - **Terminal cost**: zero
# - **Terminal constraint**: reach L2  (ψ(xN) = xN - Lp2 = 0)
# - **No path constraints** (could add Jacobi constant, etc.)

ϵ = 1e-16
stage = StageCost((x, u, t) -> sqrt(u' * u + ϵ))
term  = TerminalCost(x -> zero(eltype(x)))

# Terminal constraint: final state = L2
ψ(x) = x - SVector{6}(Lp2)
tc = TerminalConstraint(ψ, 6)

prob = DDPProblem(flow, stage, term, nx, nu; terminal_eq=tc)

# ## Solve with DDP (iLQR)
sol = DDP.solve(prob, X0, U0, t;
	opts = DDPOptions(
		method     = :iLQR,
		max_iter   = 300,
		max_outer  = 40,
		atol       = 1e-10,
		rtol       = 1e-8,
		ctol       = 1e-6,
		μ0         = 1.0,
		ϕ_μ        = 10.0,
		μ_max      = 1e8,
		reg0       = 1e-6,
		verbose    = true,
	))

println("Status: ", sol.status)
println("Cost:   ", sol.J)
println("Terminal error: ", norm(sol.X[end] - SVector{6}(Lp2)))

# ## Plot the solution
p = plot_xy(μ)
scatter!(p, [x[1] for x in sol.X], [x[2] for x in sol.X],
         ms = 2, label = false, color = :green)

for k in 1:(N-1)
	xk = sol.X[k]
	uk = sol.U[k]
	x0seg = xk + B_imp * uk
	solseg = Motion.CR3BP.build_solution(μ, x0seg, sol.t[k], sol.t[k+1],
	                                      Vern9(); reltol = 1e-8, abstol = 1e-8)
	τ = range(sol.t[k], sol.t[k+1], length = 100)
	Xseg = reduce(hcat, solseg.(τ))
	plot!(p, Xseg[1, :], Xseg[2, :], label = false, linewidth = 0.75,
	      color = :grey, style = :dash)
end
p
