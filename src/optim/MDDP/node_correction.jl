# ──────────────────────────────────────────────────────────────────────
#  Node correction – update multiple-shooting initial conditions
#
#  After running DDP on each leg, we have value-function sensitivities
#  (Vx₀ᵐ, Vxx₀ᵐ) at each leg's initial state.  The node-correction
#  step updates the shooting-node states to reduce the defects
#  (continuity violations) between consecutive legs.
#
#  Defect at node m:
#      dₘ = X₁ᵐ⁺¹ - X_{Nₘ}ᵐ   (initial of next leg minus final of current)
#
#  We form a block-tridiagonal system from the linearised optimality
#  conditions (value-function Hessians + defect penalty + STMs) and
#  solve for corrections δx₂, …, δxₘ to the free shooting nodes.
#
#  Reference: Pellegrini & Russell, Acta Astronautica 2020, §4.
# ──────────────────────────────────────────────────────────────────────

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

Newton-like correction of the shooting-node initial conditions to
reduce continuity defects.

The system minimises over free-node corrections δx₂, …, δxₘ:

    ∑ₘ [½ δxₘᵀ Vxx₀ᵐ δxₘ + Vx₀ᵐ δxₘ]
    + (μ_defect/2) ∑ₘ ‖dₘ + Φₘ δxₘ - δxₘ₊₁‖²

where Φₘ is the state-transition matrix from leg-start to leg-end,
computed by chaining the linearised dynamics along each leg.

After solving, updates `legs[m].X[1]` for m = 2, …, M and
re-propagates each affected leg with existing controls.
"""
function node_correction!(legs::Vector{<:Leg}, Vx0s, Vxx0s,
                          prob::MDDPProblem, μ_defect)
    M = length(legs)
    nx = prob.nx
    T = eltype(legs[1].X[1])

    M == 1 && return  # nothing to correct with a single leg

    # ── Compute STMs: Φₘ = ∂X_{Nₘ}ᵐ / ∂X₁ᵐ ───────────────────────
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

    # ── Compute defects ─────────────────────────────────────────────
    defects_vec = compute_defects(legs)

    # ── Assemble block-tridiagonal Newton system ────────────────────
    # Decision variables: δx₂, δx₃, …, δxₘ  (M-1 free nodes)
    n_nodes = M - 1
    n_total = nx * n_nodes

    H = zeros(T, n_total, n_total)
    g = zeros(T, n_total)

    Inx = Matrix{T}(LinearAlgebra.I, nx, nx)

    for m in 1:n_nodes
        # Block indices for node m+1 (legs[m+1])
        ri = ((m - 1) * nx + 1):(m * nx)

        # Value function Hessian + gradient from leg m+1
        @views H[ri, ri] .+= Vxx0s[m+1]
        @views g[ri]     .+= Vx0s[m+1]

        # Defect penalty: (μ/2) ‖dₘ + δxₘ₊₁ - Φₘ δxₘ‖²
        Φm = STMs[m]
        dm  = defects_vec[m]

        # ∂²/∂δxₘ₊₁² contribution
        @views H[ri, ri] .+= μ_defect .* Inx

        if m > 1
            rj = ((m - 2) * nx + 1):((m - 1) * nx)
            ΦmM = Matrix(Φm)  # dense for block multiply into H
            # ∂²/∂δxₘ² from Φₘᵀ Φₘ
            @views H[rj, rj] .+= μ_defect .* (ΦmM' * ΦmM)
            # Cross terms
            @views H[ri, rj] .+= -μ_defect .* ΦmM'
            @views H[rj, ri] .+= -μ_defect .* ΦmM
            # Gradient w.r.t. δxₘ from defect m
            @views g[rj] .+= -μ_defect .* (Φm' * dm)
        end

        # Gradient w.r.t. δxₘ₊₁ from defect m
        @views g[ri] .+= μ_defect .* dm
    end

    # Regularise and solve
    for i in 1:n_total
        H[i, i] += T(1e-8)
    end
    δx_all = -(H \ g)

    # ── Apply corrections and re-propagate ──────────────────────────
    for m in 1:n_nodes
        i1 = (m - 1) * nx + 1
        i2 = m * nx
        δx = SVector{nx,T}(δx_all[i1:i2])
        legs[m+1].X[1] = legs[m+1].X[1] + δx
        _repropagate!(legs[m+1], prob)
    end
end

"""
    _repropagate!(leg, prob)

Re-propagate states on a leg using existing controls, starting from
the (updated) initial state `leg.X[1]`.
"""
function _repropagate!(leg::Leg, prob::MDDPProblem)
    N = length(leg.X)
    for k in 1:(N-1)
        leg.X[k+1] = prob.dynamics(leg.X[k], leg.U[k], leg.t[k], leg.t[k+1])
    end
end
