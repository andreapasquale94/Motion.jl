# # CR3BP: Connecting L1 → L2 Manifolds with Impulsive Multiple Shooting
#
# This tutorial demonstrates a practical workflow to connect two CR3BP manifolds
# (unstable from L1 and stable into L2) using a *multiple shooting* formulation
# with impulsive corrections at the nodes.
#
# We will:
# 1. Compute approximate unstable/stable manifold trajectories near L1/L2.
# 2. Sample points at equal "stretch" (arc-length-like) increments.
# 3. Build an initial guess for a multiple shooting problem.
# 4. Solve a constrained NLP with `Optimization.jl` + `Ipopt`.
# 5. Plot the solution.
#
# **Deps**
# - `Motion` (your package with CR3BP utilities)
# - `OrdinaryDiffEqVerner` for Vern9 etc.
# - `Optimization`, `OptimizationIpopt` for optimisation
# - `ForwardDiff` for gradients computation
# - `Plots` for plotting
#
# The code is written to be compatible with **Literate.jl**: run it as a normal Julia
# script, or convert it to Markdown/Notebook with Literate.

using Motion
using LinearAlgebra
using OrdinaryDiffEqVerner
using Plots
using StaticArrays

using Optimization, OptimizationIpopt
using ForwardDiff

# ## Problem setup
const μ = 0.012150584269940356

# A small plot helper: primaries + L1/L2.
function plot_xy(μ)
    lps = Motion.libration_points(μ)[1:2]
    p = plot(framestyle = :box, xlabel = "x (-)", ylabel = "y (-)", aspect_ratio = 1)
    scatter!([1-μ], [0], label=false, marker=:o, color=:grey)
    for i in eachindex(lps)
        scatter!(p, [lps[i][1]], [lps[i][2]], label=false, marker=:d, color=:red)
    end
    return p
end

# ## Approximate manifolds near L1 / L2
#
# We compute a direction from the Jacobian eigenvectors and integrate forward/backward.
# This is *not* a full manifold computation, but good enough to seed a multiple-shooting initial guess.

LPs = Motion.libration_points(μ)

# Unstable (forward in time) from L1
tfu = 1.2π
Lp1 = LPs[1]
M1 = Motion.CR3BP.jacobian(Lp1, μ)
eig1 = eigen(M1)
w_u = real(eig1.vectors[:, end])  # heuristic: take last eigenvector

x0u = Lp1 .+ 1e-3 * w_u
sol_u = Motion.CR3BP.build_solution(μ, x0u, 0.0, tfu, Vern9(); abstol=1e-12, reltol=1e-12);
dt_u = LinRange(0, tfu, 1000)
X_wu = reduce(hcat, sol_u.(dt_u));

# Stable (backward in time) into L2
tsu = -2π
Lp2 = LPs[2]
M2 = Motion.CR3BP.jacobian(Lp2, μ)
eig2 = eigen(M2)
w_s = real(eig2.vectors[:, end])

x0s = Lp2 .+ 1e-2 * w_s
sol_s = Motion.CR3BP.build_solution(μ, x0s, 0.0, tsu, Vern9(); abstol=1e-12, reltol=1e-12)
dt_s = LinRange(0, tsu, 1000)
X_ws = reduce(hcat, sol_s.(dt_s))

p0 = plot_xy(μ)
plot!(p0, X_wu[1,:], X_wu[2,:], color=:red,   style=:dot, label="\$\\mathcal{W}_u\$ (approx)")
plot!(p0, X_ws[1,:], X_ws[2,:], color=:green, style=:dot, label="\$\\mathcal{W}_s\$ (approx)")
p0

# ## Sample nodes using a stretch measure
#
# We use `Motion.compute_stretch` to create node times such that successive samples have approximately 
# constant stretching measure increment `Δs`. This is handy to distribute nodes along the curve.

Δs = 0.1

sol_w_u = Motion.compute_stretch(sol_u, 0, tfu,  Δs, Vern9())
X_wus = reduce(hcat, sol_u.(sol_w_u.t))

sol_w_s = Motion.compute_stretch(sol_s, 0, tsu, -Δs, Vern9())
X_wss = reduce(hcat, sol_s.(sol_w_s.t))

p1 = plot_xy(μ)
plot!(p1, X_wu[1,:], X_wu[2,:], color=:red, style=:dot, label="\$\\mathcal{W}_u\$")
scatter!(p1, X_wus[1,:], X_wus[2,:], color=:black, ms=2, label=false)
plot!(p1, X_ws[1,:], X_ws[2,:], color=:green, style=:dot, label="\$\\mathcal{W}_s\$")
scatter!(p1, X_wss[1,:], X_wss[2,:], color=:black, ms=2, label=false)
p1

# ## Build initial guess
#
# Layout in decision vector:
# - `t0` (scalar)
# - `dt` (N - 1)
# - `X` (nx × N)
# - `U` (nu × N)
#
# Here: `U` are *impulses* applied at each node: we model them as an instantaneous velocity kick.

t00 = 0.0

dt0 = vcat(diff(sol_w_u.t), -reverse(diff(sol_w_s.t)))
dt0[1]  = 0.5
dt0[end] = 0.5

X0 = reduce(hcat, vcat(
    sol_u.(sol_w_u.t[1:(end-1)]),   # exclude last to avoid duplication
    reverse(sol_s.(sol_w_s.t))
))

nx, N = size(X0)
nu = 3
U0 = 1e-9 * ones(nu, N)  # small impulses as initial guess

xx0 = vcat(t00, dt0, reshape(X0, nx*N), reshape(U0, nu*N))

vN  = Val(N)
vnx = Val(nx)
vnu = Val(nu)
vobj = Val(:FUEL);

# ## Define the flow, objective flow and constraints
#
# We define `flow(x,u,t0,t1)` as:
# 1) apply instantaneous kick `x0 = x + B*u`
# 2) propagate under CR3BP dynamics to next node.

B = zeros(nx, nu)
B[(nx-nu+1):end, :] .= I(nu)  # add to velocity components only

flow = (x, u, t0, t1) -> begin
    x0 = x + B*u
    Motion.CR3BP.flow(μ, x0, t0, t1, Vern9(); reltol=1e-8, abstol=1e-8)
end

objective = (x, p) -> Motion.ImpulsiveShooting.objective(x, vN, vnx, vnu, vobj)

constraints = (out, x, p) -> begin
    out .= Motion.ImpulsiveShooting.defects(x, flow, vN, vnx, vnu)
    nothing
end;

# Bounds: constrain dt positive, and fix endpoint states to L1 and L2.
it0, idt, iX, iU = Motion.ImpulsiveShooting.indexes(vN, vnx, vnu)

nvars = length(xx0)
lb = fill(-Inf, nvars)
ub = fill( Inf, nvars)
lb[idt] .= 1e-12
ub[idt] .= 1.0;

# fix initial point
lb[iX[:,1]] .= LPs[1]
ub[iX[:,1]] .= LPs[1];

# fix final point
lb[iX[:,end]] .= LPs[2]
ub[iX[:,end]] .= LPs[2];

# Optimization function
optfunc = OptimizationFunction(objective, AutoForwardDiff(); cons = constraints);

# Create optimization problem
clb = fill(0.0, nx*(N-1))
cub = fill(0.0, nx*(N-1))

prob = OptimizationProblem(optfunc, xx0;
    lb = lb, ub = ub,
    lcons = clb, ucons = cub,
);

# ## Solve the NLP
sol = solve(prob,
    IpoptOptimizer(
        hessian_approximation = "limited-memory",
        limited_memory_max_history = 10,
    );
    maxiters = 1000,
    verbose = 4,
    xtol = 1e-4,
);

# ## Plot the solution
#
# We unpack `(X,U,t)` from the solution and then integrate each segment using
# the optimized impulse `u_k` and node times.

_, _, Xo, Uo, to = Motion.ImpulsiveShooting.variables(sol.u, vN, vnx, vnu)

p2 = plot_xy(μ)
scatter!(p2, Xo[1,:], Xo[2,:], ms=2, label=false, color=:green)

for k in 1:(N-1)
    xk = @view Xo[:,k]
    uk = @view Uo[:,k]

    x0seg = xk + B*uk
    solseg = Motion.CR3BP.build_solution(μ, x0seg, to[k], to[k+1], Vern9(); reltol=1e-8, abstol=1e-8)

    τ = range(to[k], to[k+1], length=100)
    Xseg = reduce(hcat, solseg.(τ))

    plot!(p2, Xseg[1,:], Xseg[2,:], label=false, linewidth=0.75, color=:grey, style=:dash)
end
p2