# # CR3BP: Halo Orbit Seeding from a Lyapunov Bifurcation
#
# **Halo orbits** are three-dimensional periodic orbits that bifurcate from the
# planar Lyapunov family when a stability index crosses the critical value of ±2.
# At the bifurcation point the monodromy matrix acquires a pair of eigenvalues
# at +1 (in addition to the trivial pair), and the corresponding eigenvector
# points *out of the orbital plane* — it is the direction in which the new
# three-dimensional family is born.
#
# This tutorial demonstrates how to:
#
# 1. Identify the **bifurcation point** along the Lyapunov family from the
#    stability indices computed in the previous tutorial,
# 2. Extract the **bifurcating eigenvector** from the monodromy matrix,
# 3. Build a 3D seed by displacing the Lyapunov orbit along that eigenvector,
# 4. Correct the seed into a periodic halo orbit with a single-shooting corrector.
#
# The corrected orbit will serve as the starting point for halo family
# continuation in the next tutorial.

using LinearAlgebra
using Serialization
using OrdinaryDiffEqVerner
using StaticArrays
using Plots

using Motion
using Motion.Continuation

# ## Load the Lyapunov family data
#
# We load the family from the pseudo-arclength continuation tutorial.
const data = deserialize(joinpath(@__DIR__, "cache", "021_L1_Lyap.jls"))
const μ = data.μ

X = reduce(hcat, [d.x for d in data.data])
T = [d.T for d in data.data]

# ## Identify the bifurcation orbit
#
# We compute the stability indices for each orbit and look for the point where
# `p` crosses −2 — this is where the halo family branches off. We detect this
# by checking for a sign change in `(p + 2)` between consecutive orbits.
M = zeros(length(T), 6, 6)
s = zeros(length(T), 2)

for i in eachindex(T)
    M[i, :, :] = Motion.CR3BP.monodromy_matrix(μ, X[:, i], 0.0, 2T[i], Vern9())
    s[i, :] .= Motion.CR3BP.stability_index(M[i, :, :])
end

bif_idx = findfirst(i -> (s[i, 1] + 2) * (s[i-1, 1] + 2) < 0, 2:length(T)) + 1

# ## Build the halo seed
#
# At the bifurcation point, we compute the eigenvectors of the monodromy matrix.
# The eigenvector associated with the bifurcating eigenvalue (near +1) has a
# non-zero out-of-plane component — this is the direction in which we displace
# the Lyapunov orbit to create a 3D halo seed.
L, W = eigen(M[bif_idx, :, :])

vi = real(W[:, 2] / norm(W[:, 2]))

# Displace the Lyapunov orbit by a small amplitude `ε` along the bifurcating
# eigenvector, then enforce the **xz-plane symmetry** of halo orbits:
# - start on the xz-plane: `y = 0`
# - perpendicular crossing: `vₓ = vz = 0`
# - free components: `x`, `z`, `vy`
x0g = X[:, bif_idx] + 1e-4 * vi
x0 = [x0g[1], 0, x0g[3], 0, x0g[5], 0]
T0 = 2T[bif_idx]

# ## Single-shooting correction
#
# Halo orbits have three free state components (`x`, `z`, `vy`) plus the period,
# giving a reduced vector `z = [x, z, vy, T]` of dimension 4. The full-period
# **periodicity constraint** provides 3 equations (one per free component).
# A **natural-parameter constraint** on `z` (index 2 in the reduced vector)
# closes the system.
layout = SingleShootingReducedLayout(6, [1, 3, 5], true);

z0 = [x0[1], x0[3], x0[5], T0]

f(x, T, λ) = Motion.CR3BP.flow(μ, x, 0.0, T, Vern9(); abstol =  reltol=1e-14 );
sr = SingleShootingResidual(
	SingleShooting(f, layout), Continuation.Periodicity(layout),
);

# Pin `z` (the out-of-plane position) to its initial-guess value:
r = NaturalParameterShootingResidual(sr, 2);

# Solve the corrector:
corr = Continuation.SciMLCorrector(; abstol =  reltol=1e-10 , verbose = false);
zsol, stat = solve(r, corr, z0, z0[2]);

xn, Tn = Continuation.unpack(layout, zsol)

# ## Plot the seed vs the corrected halo orbit

# Integrate the initial seed:
sol_guess = Motion.CR3BP.build_solution(μ, x0, 0.0, T0, Vern9(); abstol =  reltol=1e-14 );
X_guess = reduce(hcat, sol_guess.(LinRange(0, T0, 1000)));

# Integrate the corrected orbit:
sol = Motion.CR3BP.build_solution(μ, xn, 0.0, Tn, Vern9(); abstol =  reltol=1e-14 )
X_corr = reduce(hcat, sol.(LinRange(0, Tn, 1000)));

begin
	p = plot(framestyle = :box, xlabel = "x (-)", ylabel = "z (-)", aspect_ratio = 1, legend = :bottomright)
	plot!(p, X_guess[1, :], X_guess[3, :], color = :red, style = :dot, label = "Initial guess")
	plot!(p, X_corr[1, :], X_corr[3, :], color = :black, label = "Corrected halo orbit")
end

## hideall
cache_path = joinpath(@__DIR__, "cache", "031_L1_SHalo_seed.jls")
	mkpath(dirname(cache_path))
	serialize(cache_path,
		(; μ = μ, x = xn, T = Tn )
) # hide
