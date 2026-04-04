# ──────────────────────────────────────────────────────────────────────
#  Node correction – update multiple-shooting initial conditions
#
#  After running DDP on each leg, we have value-function gradients
#  Vx₀ᵐ at each leg's initial state.  The node-correction step
#  updates the shooting-node states to reduce the defects
#  (continuity violations) between consecutive legs.
#
#  The defect at node m is:
#      dₘ = X₁ᵐ⁺¹ - Xₙₘᵐ   (initial state of next leg minus
#                              final propagated state of current leg)
#
#  We linearise and solve the coupled system via a block-tridiagonal
#  Newton step.
#
#  Reference: Pellegrini & Russell, Acta Astronautica 2020, §4.
# ──────────────────────────────────────────────────────────────────────

using ..DDP: dynamics_derivatives

"""
    compute_defects(legs) -> Vector{SVector}

Compute the continuity defects between consecutive legs:
    dₘ = X₁ᵐ⁺¹ - X_{Nₘ}ᵐ   for m = 1, …, M-1
"""
function compute_defects(legs::Vector{<:Leg})
    M = length(legs)
    defects = Vector{typeof(legs[1].X[1])}(undef, M - 1)
    for m in 1:(M-1)
        defects[m] = legs[m+1].X[1] - legs[m].X[end]
    end
    return defects
end

"""
    max_defect(legs) -> T

Maximum absolute defect across all shooting nodes.
"""
function max_defect(legs::Vector{<:Leg})
    M = length(legs)
    T = eltype(legs[1].X[1])
    d_max = zero(T)
    for m in 1:(M-1)
        d_max = max(d_max, maximum(abs, legs[m+1].X[1] - legs[m].X[end]))
    end
    return d_max
end

"""
    node_correction!(legs, Vx0s, Vxx0s, prob, μ_defect)

Perform a Newton-like correction of the shooting-node initial conditions
to reduce continuity defects.

Uses the value-function Hessians from the DDP backward passes to form
a block-tridiagonal system.  The penalty `μ_defect` weights the defect
penalty in the merit function.

The correction updates `legs[m].X[1]` for m = 2, …, M and re-propagates
each affected leg's trajectory using the existing controls.
"""
function node_correction!(legs::Vector{<:Leg}, Vx0s, Vxx0s,
                          prob::MDDPProblem, μ_defect)
    M = length(legs)
    nx = prob.nx
    T = eltype(legs[1].X[1])

    if M == 1
        return  # nothing to correct
    end

    # Compute STMs from initial to final state of each leg
    # Φₘ = ∂X_{Nₘ}ᵐ / ∂X₁ᵐ  (accumulated via chain rule)
    STMs = Vector{SMatrix{nx,nx,T}}(undef, M)
    for m in 1:M
        Nm = length(legs[m].X)
        Φ = SMatrix{nx,nx,T}(I)
        for k in 1:(Nm-1)
            fx, _ = dynamics_derivatives(
                prob.dynamics, legs[m].X[k], legs[m].U[k],
                legs[m].t[k], legs[m].t[k+1])
            Φ = fx * Φ
        end
        STMs[m] = Φ
    end

    # Compute defects
    defects_vec = compute_defects(legs)

    # Build and solve the block-tridiagonal system
    # Decision variables: δx₂, δx₃, …, δxₘ  (corrections to nodes 2..M)
    # The system enforces:
    #   Vxx₀ᵐ δxₘ + Vx₀ᵐ + μ_defect * Φₘ₋₁ᵀ(Φₘ₋₁ δxₘ₋₁ - δxₘ + dₘ₋₁)
    #                     - μ_defect * (δxₘ - Φₘ₋₁ δxₘ₋₁ - dₘ₋₁) = 0
    #
    # Simplified: we solve the least-squares defect correction
    #   minimise ∑ₘ [½ δxₘᵀ Vxx₀ᵐ δxₘ + Vx₀ᵐ δxₘ + (μ/2)‖dₘ + Φₘ δxₘ - δxₘ₊₁‖²]
    #
    # KKT gives a block-tridiagonal system.

    n_nodes = M - 1  # number of free nodes (node 1 is fixed)
    n_total = nx * n_nodes

    # Assemble dense system (for moderate M; for large M a banded solver is better)
    H = zeros(T, n_total, n_total)
    g = zeros(T, n_total)

    for m in 1:n_nodes
        # Block indices for node m+1 (legs[m+1])
        i1 = (m - 1) * nx + 1
        i2 = m * nx

        # Value function Hessian contribution from leg m+1
        Vxx = Matrix(Vxx0s[m+1])
        H[i1:i2, i1:i2] += Vxx

        # Value function gradient from leg m+1
        g[i1:i2] += Vector(Vx0s[m+1])

        # Defect penalty: dₘ = X₁ᵐ⁺¹ - X_{Nₘ}ᵐ
        # After correction: d̃ₘ = (X₁ᵐ⁺¹ + δxₘ₊₁) - Xf(X₁ᵐ + δxₘ)
        #                      ≈ dₘ + δxₘ₊₁ - Φₘ δxₘ
        # Penalty: (μ/2) ‖dₘ + δxₘ₊₁ - Φₘ δxₘ‖²
        Φm = Matrix(STMs[m])
        dm = Vector(defects_vec[m])

        # ∂²/∂δxₘ₊₁² = μ I
        H[i1:i2, i1:i2] += μ_defect * I(nx)

        # ∂²/∂δxₘ² (from Φₘᵀ Φₘ) – only if m > 1 (node m = legs[m], δxₘ exists)
        if m > 1
            j1 = (m - 2) * nx + 1
            j2 = (m - 1) * nx
            H[j1:j2, j1:j2] += μ_defect * Φm' * Φm
            # Cross term: ∂²/∂δxₘ ∂δxₘ₊₁ = -μ Φₘ
            H[i1:i2, j1:j2] += -μ_defect * Φm'
            H[j1:j2, i1:i2] += -μ_defect * Φm
            # Gradient w.r.t. δxₘ from defect m
            g[j1:j2] += -μ_defect * Φm' * dm
        end

        # Gradient w.r.t. δxₘ₊₁ from defect m
        g[i1:i2] += μ_defect * dm
    end

    # Also add defect penalty contribution from leg M's STM on last node
    # (defect M-1 involves Φₘ₋₁ applied to the last free node)

    # Regularise and solve
    H_reg = H + T(1e-8) * I(n_total)
    δx_all = -H_reg \ g

    # Apply corrections and re-propagate
    for m in 1:n_nodes
        i1 = (m - 1) * nx + 1
        i2 = m * nx
        δx = SVector{nx,T}(δx_all[i1:i2])
        legs[m+1].X[1] = legs[m+1].X[1] + δx

        # Re-propagate leg m+1 with existing controls
        _repropagate!(legs[m+1], prob)
    end
end

"""
    _repropagate!(leg, prob)

Re-propagate the states on a leg using the existing controls,
starting from the (updated) initial state `leg.X[1]`.
"""
function _repropagate!(leg::Leg, prob::MDDPProblem)
    N = length(leg.X)
    for k in 1:(N-1)
        leg.X[k+1] = prob.dynamics(leg.X[k], leg.U[k], leg.t[k], leg.t[k+1])
    end
end
