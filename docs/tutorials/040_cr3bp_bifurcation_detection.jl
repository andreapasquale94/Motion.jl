# # CR3BP: Bifurcation Detection & Broucke Diagrams
#
# Periodic orbit families in the CR3BP are parameterised by a continuous quantity
# (Jacobi constant, arc-length, …).  As one traces a family, the **stability
# indices** `ν₁` and `ν₂` — scalar summaries of the monodromy matrix —
# evolve continuously.  Whenever an index crosses a **critical value**, a new
# orbit family is born: this is a **bifurcation**.
#
# The table below summarises the most important bifurcation types:
#
# | Type              | Critical `ν` | Eigenvalue        | New-family period |
# |-------------------|:------------:|:-----------------:|:-----------------:|
# | Tangent           |  `+2`        | `+1`              | `T`   (same)      |
# | Period-doubling   |  `-2`        | `-1`              | `2T`              |
# | Period-tripling   |  `-1`        | `e^{±2πi/3}`      | `3T`              |
# | Period-quadrupling|   `0`        | `±i`              | `4T`              |
# | General period-*N*| `2cos(2π/N)` | `e^{±2πi/N}`     | `NT`              |
#
# The **Broucke diagram** is the standard visualisation: it plots `ν₁` and
# `ν₂` against a family parameter, with horizontal reference lines at the
# critical values.  Crossings of those lines are the bifurcation points.
#
# This tutorial demonstrates how to:
#
# 1. Compute the Broucke diagram for the Earth–Moon L₁ Lyapunov family,
# 2. Automatically **detect** all standard bifurcations,
# 3. **Locate** the Lyapunov → Halo bifurcation to high accuracy,
# 4. **Exploit** the bifurcation to generate a halo-orbit seed,
# 5. Correct the seed into a periodic halo orbit.
#
# ## Theory refresher
#
# Given the 6×6 monodromy matrix `M` of a periodic orbit (the STM evaluated
# over one full period), one pair of eigenvalues is always `(1, 1)` — the
# trivial pair from the Jacobi-constant integral.  The remaining four
# eigenvalues come in two reciprocal pairs `(λ, 1/λ)`.  Each pair is
# captured by a single scalar:
#
# ```math
# \nu = \lambda + \frac{1}{\lambda}
# ```
#
# which is the **Broucke stability index**.  The orbit is linearly stable in
# the corresponding direction when `|ν| < 2` (eigenvalues on the unit circle)
# and unstable when `|ν| > 2` (real eigenvalues off the unit circle).
#
# A bifurcation occurs when `ν` passes through a critical value.  For the
# Lyapunov → Halo transition, the **out-of-plane** index `ν₁` crosses `+2`,
# meaning an eigenvalue passes through `+1` — a **tangent bifurcation**.

using LinearAlgebra
using Serialization
using OrdinaryDiffEqVerner
using StaticArrays
using Plots

using Motion
using Motion.Continuation

# ## Load the Lyapunov family
#
# We load the family computed by the pseudo-arclength continuation tutorial.
# Each entry contains the initial state `x` and the half-period `T`.
const data = deserialize(joinpath(@__DIR__, "cache", "021_L1_Lyap.jls"))
const μ = data.μ

X = reduce(hcat, [d.x for d in data.data])
T = [d.T for d in data.data]

# ## Step 1 — Build the Broucke diagram
#
# We wrap the family as `ContinuationPoint`s and provide a monodromy function
# that integrates the STM over one full period (`2T`, since `T` is the half-period).

family = [
    ContinuationPoint{Float64}([X[1, i], X[5, i], T[i]], X[1, i])
    for i in eachindex(T)
]

function mono_fn(cp)
    idx = argmin(abs.(X[1, :] .- cp.λ))
    return Motion.CR3BP.monodromy_matrix(μ, X[:, idx], 0.0, 2T[idx], Vern9())
end

# The `broucke_diagram` function computes the stability indices at every family
# member and optionally runs all requested detectors to find bifurcation
# crossings in a single pass.
bd = broucke_diagram(
    family, mono_fn;
    parameter_fn = cp -> cp.λ,
    detectors    = ALL_STANDARD_DETECTORS,
)

# ## Step 2 — Plot the Broucke diagram
#
# The resulting `BrouckeData` object contains everything we need for the plot:
# the parameter vector, the two stability index vectors, and the detected
# bifurcation events.

begin
    p = plot(
        xlabel     = "x₀ (-)",
        ylabel     = "Stability index ν",
        framestyle = :box,
        dpi        = 200,
        legend     = :topright,
        ylim       = (-5, 5),
    )

    ## Stability indices
    plot!(p, bd.parameter, bd.p; lw = 2, label = "ν₁")
    plot!(p, bd.parameter, bd.q; lw = 2, label = "ν₂")

    ## Reference lines for each standard bifurcation type
    bif_styles = [
        (2.0,  "tangent (ν = +2)"),
        (-2.0, "period-doubling (ν = -2)"),
        (-1.0, "period-tripling (ν = -1)"),
        (0.0,  "period-quadrupling (ν = 0)"),
    ]
    colors = [:red, :blue, :green, :orange]
    for (k, (sc, lab)) in enumerate(bif_styles)
        hline!(p, [sc]; ls = :dash, lw = 1, c = colors[k], label = lab)
    end

    ## Mark detected bifurcation events
    for ev in bd.bifurcations
        x_bif = (bd.parameter[ev.index] + bd.parameter[ev.index + 1]) / 2
        scatter!(p, [x_bif], [ev.s_critical];
            marker     = :star5,
            markersize = 8,
            c          = :black,
            label      = false,
        )
    end

    p
end

# Each black star marks a crossing.  Depending on the family extent, we
# typically see the **tangent bifurcation** (ν = +2) where the halo orbits
# are born, and possibly a **period-doubling** (ν = -2) further along.

# ## Step 3 — Identify and print all detected bifurcations
#
# The `BrouckeData` stores every crossing as a `BifurcationEvent`.

for (k, ev) in enumerate(bd.bifurcations)
    x_lo = bd.parameter[ev.index]
    x_hi = bd.parameter[ev.index + 1]
    println("Event $k:  ν$(ev.which) crosses $(ev.s_critical) between " *
            "x₀ ∈ [$x_lo, $x_hi]  (period ×$(ev.period_mul))")
end

# ## Step 4 — Locate the Lyapunov → Halo bifurcation
#
# We pick the first tangent bifurcation (ν crossing +2) and refine it.
# `locate_bifurcation` uses a bisection–secant hybrid: it interpolates
# between the two bracketing family members, corrects the orbit, recomputes
# the monodromy matrix, and narrows the bracket until `|ν - 2| < tol`.

## Find the tangent bifurcation event
tang_events = filter(e -> e.s_critical ≈ 2.0, bd.bifurcations)
isempty(tang_events) && error("No tangent bifurcation found — extend the family")
tang_event = tang_events[1]

# We need a differential corrector and a monodromy function that work with
# `ContinuationPoint`s.  The corrector takes a guess `(z, λ)` and returns
# the corrected orbit.

layout_lyap = SingleShootingReducedLayout(6, [1, 5], true);
f_lyap(x, TT, λ) = Motion.CR3BP.flow(μ, x, 0.0, TT, Vern9(); abstol = 1e-14, reltol = 1e-14);
sr_lyap = SingleShootingResidual(
    SingleShooting(f_lyap, layout_lyap), Continuation.HalfPeriodSymmetry([2, 4]),
);
r_lyap = NaturalParameterShootingResidual(sr_lyap, 1);
corr_lyap = Continuation.SciMLCorrector(; abstol = 1e-12, reltol = 1e-12);

function correct_fn(z_guess, λ)
    try
        zsol, stat = solve(r_lyap, corr_lyap, z_guess, λ)
        return (zsol, stat.success)
    catch
        return (z_guess, false)
    end
end

## For monodromy, unpack the reduced state → full state → integrate STM
function monodromy_fn(cp)
    xn, Tn = Continuation.unpack(layout_lyap, cp.z)
    return Motion.CR3BP.monodromy_matrix(μ, xn, 0.0, 2Tn, Vern9())
end

## Build a refined family of ContinuationPoints for the locate function
family_lyap = [
    ContinuationPoint{Float64}([X[1, i], X[5, i], T[i]], X[1, i])
    for i in eachindex(T)
]

bp = locate_bifurcation(family_lyap, tang_event, correct_fn, monodromy_fn; tol = 1e-8)
println("Bifurcation located at x₀ = $(bp.cp.z[1])")
println("  stability indices:  ν₁ = $(bp.stability_indices[1]),  ν₂ = $(bp.stability_indices[2])")
println("  critical eigenvalue: $(bp.eigenvalue)")

# ## Step 5 — Exploit the bifurcation to seed a halo orbit
#
# At the tangent bifurcation, the monodromy matrix has an eigenvector with a
# non-zero out-of-plane component — this is the direction in which the halo
# family is born.  `exploit_bifurcation` perturbs the Lyapunov orbit by `ε`
# along this eigenvector and corrects the result.
#
# For a tangent bifurcation the new family has the **same period** as the
# parent orbit, so we can use a halo-orbit corrector with period `T`.

## Halo shooting setup: free variables are [x, z, vy, T/2]
layout_halo = SingleShootingReducedLayout(6, [1, 3, 5], true);
f_halo(x, TT, λ) = Motion.CR3BP.flow(μ, x, 0.0, TT, Vern9(); abstol = 1e-14, reltol = 1e-14);
sr_halo = SingleShootingResidual(
    SingleShooting(f_halo, layout_halo), Continuation.HalfPeriodSymmetry([2, 4, 6]),
);
r_halo = NaturalParameterShootingResidual(sr_halo, 2);
corr_halo = Continuation.SciMLCorrector(; abstol = 1e-12, reltol = 1e-12);

## Build the halo perturbation vector in halo z-space: [x, z, vy, T/2]
eigvec = real.(bp.eigenvector)
pert_halo = [eigvec[1], eigvec[3], eigvec[5], 0.0]

## Halo corrector: pin z (index 2 in reduced state) to its perturbed value
xn_bif, Tn_bif = Continuation.unpack(layout_lyap, bp.cp.z)
z0_halo = [xn_bif[1], 0.0, xn_bif[5], Tn_bif]
z0_halo .+= 1e-3 .* pert_halo ./ norm(pert_halo)

function correct_halo_fn(z_guess, λ)
    try
        zsol, stat = solve(r_halo, corr_halo, z_guess, z_guess[2])
        return (zsol, stat.success)
    catch
        return (z_guess, false)
    end
end

zsol_halo, stat_halo = solve(r_halo, corr_halo, z0_halo, z0_halo[2])

xn_halo, Tn_halo = Continuation.unpack(layout_halo, zsol_halo)

println("Halo seed corrected:")
println("  x₀ = $(xn_halo)")
println("  T  = $(2Tn_halo)")

# ## Step 6 — Visualise the result
#
# We compare the Lyapunov orbit at the bifurcation point with the newly
# seeded halo orbit.  The halo orbit has a small but non-zero out-of-plane
# amplitude — it lives just above (or below) the Lyapunov plane.

## Integrate bifurcation Lyapunov orbit
sol_lyap = Motion.CR3BP.build_solution(
    μ, xn_bif, 0.0, 2Tn_bif, Vern9(); abstol = 1e-14, reltol = 1e-14);
X_lyap = reduce(hcat, sol_lyap.(LinRange(0, 2Tn_bif, 1000)));

## Integrate halo orbit
sol_halo = Motion.CR3BP.build_solution(
    μ, xn_halo, 0.0, 2Tn_halo, Vern9(); abstol = 1e-14, reltol = 1e-14);
X_halo = reduce(hcat, sol_halo.(LinRange(0, 2Tn_halo, 1000)));

begin
    p1 = plot(framestyle = :box, xlabel = "x (-)", ylabel = "y (-)",
              aspect_ratio = 1, dpi = 200, legend = :topright)
    plot!(p1, X_lyap[1, :], X_lyap[2, :]; lw = 2, label = "Lyapunov (bifurcation)")
    plot!(p1, X_halo[1, :], X_halo[2, :]; lw = 2, ls = :dash, label = "Halo seed")

    p2 = plot(framestyle = :box, xlabel = "x (-)", ylabel = "z (-)",
              aspect_ratio = 1, dpi = 200, legend = :topright)
    plot!(p2, X_lyap[1, :], X_lyap[3, :]; lw = 2, label = "Lyapunov (z ≈ 0)")
    plot!(p2, X_halo[1, :], X_halo[3, :]; lw = 2, ls = :dash, label = "Halo seed")

    plot(p1, p2; layout = (1, 2), size = (900, 400))
end

# ## Summary
#
# The full bifurcation workflow using `Motion.Continuation`:
#
# | Step     | Function                 | Purpose                                          |
# |----------|--------------------------|--------------------------------------------------|
# | Compute  | `broucke_diagram`        | stability indices + automatic crossing detection |
# | Detect   | `detect_bifurcations`    | find all `ν`-crossings for a given bifurcation type |
# | Locate   | `locate_bifurcation`     | refine the bifurcation point via bisection       |
# | Exploit  | `exploit_bifurcation`    | perturb along the critical eigenvector + correct |
#
# The same workflow applies to **any** bifurcation type.  For period-doubling,
# the only difference is that the corrector must use period `2T` (and the
# detector looks for `ν = -2`).  For period-tripling, use `3T` and `ν = -1`,
# and so on.
