# # CR3BP: Resonant Halo Orbits
#
# In the CR3BP the primaries orbit each other with a normalized period of `2π`.
# A periodic orbit is said to be in **p:q resonance** when its period satisfies
# `T = 2π p / q`, meaning the spacecraft completes `q` revolutions for every `p`
# revolutions of the primaries. Resonant orbits are of practical interest because
# they produce repeating ground tracks and enable low-cost station-keeping and
# transfer design.
#
# This tutorial demonstrates how to:
#
# 1. Load the southern halo family computed in the previous tutorial,
# 2. Identify all family members whose period is close to a **p:q resonance**
#    `T ≈ 2π p/q` for all coprime pairs with `p, q ≤ 15`,
# 3. Refine each candidate to exact resonance with a single-shooting corrector,
# 4. Plot the resonant orbits within the family.

using LinearAlgebra
using Serialization
using OrdinaryDiffEqVerner
using StaticArrays
using Plots

using Motion
using Motion.Continuation

# ## Load the halo family
const data = deserialize(joinpath(@__DIR__, "cache", "032_L1_SHalo.jls"))
const μ = data.μ

X = reduce(hcat, [d.x for d in data.data])
T_half = [d.T for d in data.data]
T_full = 2 .* T_half

# ## Identify resonance candidates
#
# A `p:q` resonance has period `T = 2π p/q`. We enumerate all coprime pairs
# `(p, q)` with `1 ≤ p, q ≤ 20` and scan the family for sign changes in
# `T - T_res` to find the closest member. This member serves as the initial
# guess for the corrector.

candidates = Dict{Tuple{Int,Int}, Int}()

for p in 1:15, q in 1:15
    gcd(p, q) != 1 && continue
    T_target = 2π * p / q
    for i in 2:length(T_full)
        if (T_full[i] - T_target) * (T_full[i-1] - T_target) < 0
            idx = abs(T_full[i] - T_target) < abs(T_full[i-1] - T_target) ? i : i - 1
            candidates[(p, q)] = idx
            break
        end
    end
end

# ## Correct each candidate to exact resonance
#
# For each candidate we set up a single-shooting corrector with a **fixed period**
# equal to the exact resonance value `T = 2πn`. Since the period is no longer
# free, the reduced vector becomes `z = [x, z, vy]` (3 unknowns) and the
# full-period periodicity constraint supplies 3 equations — giving a square system
# that can be solved directly without an additional phase constraint.

f(x, T, λ) = Motion.CR3BP.flow(μ, x, 0.0, T, Vern9(); abstol =  reltol=1e-14 );
corr = Continuation.SciMLCorrector(; abstol =  reltol=1e-12 , verbose = false);

resonant_orbits = Dict{Tuple{Int,Int}, NamedTuple{(:x, :T), Tuple{Vector{Float64}, Float64}}}()

for ((p, q), idx) in sort(collect(candidates))
    T_target = 2π * p / q

    # Build a layout with T fixed to the exact resonance period
    layout_fixed = SingleShootingReducedLayout(6, [1, 3, 5], false; T_fixed = T_target)

    sr = SingleShootingResidual(
        SingleShooting(f, layout_fixed),
        Continuation.Periodicity(layout_fixed),
    )

    z0 = [X[1, idx], X[3, idx], X[5, idx]]
    zsol, stat = solve(sr, corr, z0, 0.0)

    if stat.success
        xn, _ = Continuation.unpack(layout_fixed, zsol)
        resonant_orbits[(p, q)] = (; x = Vector(xn), T = T_target)
    end
end

# ## Plot the resonant orbits
begin
    p = plot(
        framestyle = :box,
        xlabel = "x (-)", ylabel = "z (-)",
        aspect_ratio = 1,
        dpi = 200,
    )

    for idx in sort(collect(keys(resonant_orbits)))
        orb = resonant_orbits[idx]
        sol = Motion.CR3BP.build_solution(
            μ, orb.x, 0.0, orb.T, Vern9(); abstol =  reltol = 1e-14 ,
        )
        Xr = reduce(hcat, sol.(LinRange(0, orb.T, 1000)))
        plot!(p, Xr[1, :], Xr[3, :], label = "$(idx[1]):$(idx[2])")
    end
    p
end
