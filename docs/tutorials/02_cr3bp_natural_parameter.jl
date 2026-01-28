# # CR3BP: Natural-parameter continuation for planar Lyapunov families
#
# This tutorial demonstrates **natural-parameter continuation** for a family of planar Lyapunov-like periodic 
# orbits in the CR3BP using `Motion.Continuation` module.
#
# We will:
#
# 1. Seed a small periodic orbit near **L₁** using the Jacobian eigen-structure,
# 2. Formulate a **single-shooting residual** with a **half-period symmetry** constraint,
# 3. Run a continuation loop where the **natural parameter** is the initial x-position `x[1]`,
# 4. Plot the resulting orbit family.
#
# The file is compatible with **Literate.jl**: it can be executed as-is, or converted
# into markdown/notebook.

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

# ## Linear seed near L₁ (center direction)
#
# We move a small distance along a center eigenvector direction, then impose planar symmetry.
x0g = xLP + 0.02 * real(W[:, 5] / norm(W[:, 5]));

# Planar initial condition: y=z=vx=vz=0 and vy free (seeded from eigenvector here).
x0 = [x0g[1], 0, 0, 0, x0g[5], 0]

# Period guess from the center eigenvalue
T0 = 2π / abs(L[5])
# ## Reduced decision variables and half-period shooting
#
# For planar Lyapunov orbits we can exploit symmetry:
#
# - If we start on the x-axis (y(0)=0) with vₓ(0)=0,
#   then periodicity can be enforced by requiring at **half-period**:
#     y(T/2) = 0 and vₓ(T/2) = 0
#
# This means we can solve a 2D boundary-value condition with a low-dimensional `z`.
#
# Here we choose reduced variables:
#
#   z = [ x[1], vᵧ(0), T/2 ]
#
# and rebuild the full state as:
#
#   x_full(0) = [ z[1], 0, 0, 0, z[2], 0 ]
#   half-period = z[3]
#
# In Motion's continuation stack:
# - `SingleShootingLayout(6)` encodes a full state dimension 6
# - `VarMap(7, [1, 5, 7])` selects which entries of `[x0; T]` are *active* in the reduced vector:
#     1 → x[1]
#     5 → vᵧ(0)
#     7 → T
#
# Note: we store T/2 as our third reduced variable, so we will interpret it accordingly later.
layout = ReducedLayout(
    SingleShootingLayout(6),
    VarMap(7, [1, 5, 7]),
    vcat(x0, T0),
)

# Flow map: propagate from t=0 to t=T (or T/2, depending on the arc object)
flow = (x, T, λ) -> Motion.CR3BP.flow(
    μ, x, 0.0, T, Vern9(); abstol = reltol = 1e-12
);

# ## Residual definition
#
# `ShootingArc(flow, layout)` defines how to build a shooting segment from reduced
# variables to the full trajectory.
#
# `HalfPeriodSymmetry((2,4))` enforces the half-period conditions on components:
#
# - index 2 = y
# - index 4 = vₓ
#
# Combined into a single residual model:
sys = SingleShootingResidual(
    ShootingArc(flow, layout), HalfPeriodSymmetry((2, 4)),
)

# Initial reduced guess
#
# We set:
# - z[1] = x[1]
# - z[2] = vᵧ(0)
# - z[3] = T/2
zinit = [x0[1], x0[5], T0/2]

# ## Corrector
#
# Continuation steps use a predictor (here natural parameter) and a corrector
# (Newton) to land back on the solution manifold.
corr = Corrector(
    SimpleNewtonRaphson(); abstol = reltol = 1e-10, verbose = true
)

# ## Continuation problem
#
# Natural parameter continuation means we pick one component as a "parameter"
# and step it directly. Here, we use:
#
# `SimpleNaturalParameter(1)` → the first reduced variable, i.e. `x[1]`,
# is treated as the continuation parameter.
#
# At each step:
# - predictor proposes a new `x[1]` (and keeps the other components as previous),
# - corrector solves for the remaining unknowns so the residual is zero again.
prob = ContinuationProblem(
    sys;
    predictor = SimpleNaturalParameter(1),
    corrector = corr,
)

# ## Initialization of history
#
# The continuation stack stores a `ContinuationPoint(z, λ)`.
history = ContinuationPoint{Float64}[
    ContinuationPoint{Float64}(zinit, zinit[1])
]

# Warm-start: do a zero step to ensure the first point is corrected tightly.
push!(history, Continuation.step!(prob, history; ds = 0)[1])
popfirst!(history)

# ## Run the continuation
#
# We take small steps in x[1].
Δs = 1e-4
nsteps = 1000

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

scatter!(p, [xLP[1]], [xLP[2]], label = false, marker = :d, color = :red)

for i in 1:10:nsteps
    point = history[i]
    xn = [point.z[1], 0, 0, 0, point.z[2], 0]
    Tn = 2 * point.z[3]

    sol = Motion.CR3BP.build_solution(
        μ, xn, 0.0, Tn, Vern9(); abstol = reltol = 1e-12
    )
    X = reduce(hcat, sol.(LinRange(0, Tn, 1000)))

    plot!(p, X[1, :], X[2, :], color = :black, linewidth = 0.5, label = false)
end
end
p
