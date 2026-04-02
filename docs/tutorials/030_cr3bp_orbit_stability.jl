# # CR3BP: Orbit Stability and Bifurcation Detection
#
# Every periodic orbit in the CR3BP has an associated **monodromy matrix** — the
# state-transition matrix evaluated over one full period. Its eigenvalues
# characterize the orbit's linear stability: eigenvalues on the unit circle
# indicate neutral stability, while eigenvalues off it signal instability.
#
# For planar orbits the 6×6 monodromy matrix always has a trivial pair of unit
# eigenvalues (associated with the energy integral and the phase along the orbit).
# The remaining four eigenvalues can be condensed into two scalar **stability
# indices** `p` and `q`. An orbit is linearly stable (in the relevant direction)
# when `|p| < 2` or `|q| < 2`, and a stability index crossing ±2 signals a
# **bifurcation** — the birth of a new orbit family (e.g., halo orbits branching
# off the Lyapunov family).
#
# This tutorial demonstrates how to:
#
# 1. Compute the monodromy matrix for every member of the L₁ Lyapunov family,
# 2. Extract the stability indices `p` and `q`,
# 3. Plot them to identify bifurcation points along the family.

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

# ## Monodromy matrix and stability indices
#
# For each orbit we integrate the **state-transition matrix (STM)** alongside the
# trajectory for one full period (`2T`, since `T` is the half-period). The STM at
# the final time is the monodromy matrix `M`. From `M` we extract the two
# stability indices via `Motion.CR3BP.stability_index`, which computes:
#
# ```math
# a = 2 - \mathrm{tr}(M), \quad
# b = \tfrac{1}{2}(a^2 - \mathrm{tr}(M^2)), \quad
# p,q = \tfrac{1}{2}\bigl(a \pm \sqrt{a^2 - 4b + 8}\bigr)
# ```

M = zeros(length(T), 6, 6)
s = zeros(length(T), 2)

for i in eachindex(T)
    M[i, :, :] = Motion.CR3BP.monodromy_matrix(μ, X[:, i], 0.0, 2T[i], Vern9())
    s[i, :] .= Motion.CR3BP.stability_index(M[i, :, :])
end

# ## Stability diagram
#
# We plot the two indices against the initial x-position of each orbit. The
# dashed red lines at ±2 mark the **bifurcation thresholds**: when an index
# crosses one of these lines, a new family branches off the Lyapunov family.
# For the Earth–Moon L₁ case, the crossing of `p = -2` corresponds to the
# birth of the **halo orbit** family.
begin
    plot(xlabel="x (-)", ylabel="Stability indices", framestyle=:box, dpi=200, ylim=(-3,3))
    plot!(X[1, :], s[:, 1], label="p", lw=2)
    plot!(X[1, :], s[:, 2], label="q", lw=2)
    hline!([-2, 2], color=:red, linestyle=:dash, label="")
end
