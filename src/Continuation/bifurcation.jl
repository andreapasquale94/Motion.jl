
# ──────────────────────────────────────────────────────────────────────
#  Bifurcation detection & exploitation for periodic-orbit continuation
# ──────────────────────────────────────────────────────────────────────

# ── Bifurcation types ─────────────────────────────────────────────────

"""
    BifurcationType

Classification of bifurcations in periodic orbit families based on the
critical value of the Broucke stability index.
"""
@enum BifurcationType begin
    TANGENT
    PERIOD_DOUBLING
    PERIOD_TRIPLING
end

"""
    critical_stability_index(bt::BifurcationType) -> Float64
    critical_stability_index(N::Integer) -> Float64

Critical value of the Broucke stability index for a given bifurcation.

For a general period-`N` bifurcation the critical value is `2cos(2π/N)`.
Special cases:
- `TANGENT`         → `+2` (eigenvalue at +1, same-period family branches off)
- `PERIOD_DOUBLING` → `-2` (eigenvalue at -1, double-period family)
- `PERIOD_TRIPLING` → `-1` (eigenvalues at `exp(±2πi/3)`)
"""
critical_stability_index(bt::BifurcationType) = _csi(Val(bt))
_csi(::Val{TANGENT})         =  2.0
_csi(::Val{PERIOD_DOUBLING}) = -2.0
_csi(::Val{PERIOD_TRIPLING}) = -1.0

function critical_stability_index(N::Integer)
    N >= 1 || throw(ArgumentError("N must be ≥ 1, got $N"))
    return 2 * cospi(2 / N)
end

"""
    period_multiple(bt::BifurcationType) -> Int
    period_multiple(N::Integer) -> Int

Period multiplier of the family emerging from the bifurcation.
"""
period_multiple(bt::BifurcationType) = _pm(Val(bt))
_pm(::Val{TANGENT})         = 1
_pm(::Val{PERIOD_DOUBLING}) = 2
_pm(::Val{PERIOD_TRIPLING}) = 3
period_multiple(N::Integer) = Int(N)

# ── Detector configuration ────────────────────────────────────────────

"""
    BifurcationDetector(bt::BifurcationType)
    BifurcationDetector(N::Integer)
    BifurcationDetector(s_critical, period_mul)

Configuration object that specifies which bifurcation to look for.

# Fields
- `s_critical::Float64` – critical stability-index value.
- `period_mul::Int`     – period multiplier of the new family.
"""
struct BifurcationDetector
    s_critical::Float64
    period_mul::Int
end

BifurcationDetector(bt::BifurcationType) =
    BifurcationDetector(critical_stability_index(bt), period_multiple(bt))

BifurcationDetector(N::Integer) =
    BifurcationDetector(critical_stability_index(N), period_multiple(N))

# ── Detection result ──────────────────────────────────────────────────

"""
    BifurcationEvent{T}

Records a sign change in `(s − s_critical)` between two consecutive
family members.

# Fields
- `index`      – index of the first bracketing point (crossing is
                  between `index` and `index + 1`).
- `s_before`   – stability index at point `index`.
- `s_after`    – stability index at point `index + 1`.
- `s_critical` – critical stability-index value.
- `which`      – which index triggered the event (`:p` or `:q`).
- `period_mul` – period multiplier of the new family.
"""
struct BifurcationEvent{T}
    index::Int
    s_before::T
    s_after::T
    s_critical::T
    which::Symbol
    period_mul::Int
end

"""
    detect_bifurcations(si, detector) -> Vector{BifurcationEvent}

Scan a vector of `(p, q)` stability-index pairs for crossings of
`detector.s_critical`.  Returns one [`BifurcationEvent`](@ref) per
detected sign change.
"""
function detect_bifurcations(
    si::AbstractVector{<:Tuple{T,T}},
    det::BifurcationDetector,
) where {T}
    events = BifurcationEvent{T}[]
    sc = T(det.s_critical)
    for k in 1:length(si)-1
        p1, q1 = si[k]
        p2, q2 = si[k+1]
        if (p1 - sc) * (p2 - sc) < 0
            push!(events, BifurcationEvent(k, p1, p2, sc, :p, det.period_mul))
        end
        if (q1 - sc) * (q2 - sc) < 0
            push!(events, BifurcationEvent(k, q1, q2, sc, :q, det.period_mul))
        end
    end
    return events
end

# ── Refined bifurcation point ─────────────────────────────────────────

"""
    BifurcationPoint{T}

A refined bifurcation point together with the critical eigenpair of its
monodromy matrix.

# Fields
- `cp`                – the [`ContinuationPoint`](@ref) at the bifurcation.
- `monodromy`         – 6×6 monodromy matrix.
- `stability_indices` – `(p, q)` stability indices.
- `eigenvalue`        – the critical eigenvalue.
- `eigenvector`       – the associated eigenvector (length 6, complex).
- `period_mul`        – period multiplier of the new family.
"""
struct BifurcationPoint{T}
    cp::ContinuationPoint{T}
    monodromy::Matrix{T}
    stability_indices::Tuple{T,T}
    eigenvalue::Complex{T}
    eigenvector::Vector{Complex{T}}
    period_mul::Int
end

# ── Stability index (local copy, model-agnostic) ─────────────────────

"""
    stability_index(M::AbstractMatrix) -> (p, q)

Broucke stability indices of a 6×6 symplectic monodromy matrix.

Given the two non-trivial eigenvalue pairs `(λ, 1/λ)`, the stability
indices satisfy `λ² − s λ + 1 = 0`.  A pair is *stable* when `|s| < 2`
and a bifurcation occurs when `s` crosses a critical value.
"""
function stability_index(M::AbstractMatrix)
    trM = tr(M)
    a  = 2 - trM
    a2 = a * a
    b  = (a2 - tr(M * M)) / 2

    disc = a2 - 4b + 8
    s = sqrt(max(disc, 0.0))          # guard against tiny negative disc
    p = (a + s) / 2
    q = (a - s) / 2
    return (p, q)
end

# ── Locate (bisection / secant) ──────────────────────────────────────

"""
    locate_bifurcation(family, event, correct_fn, monodromy_fn;
                       tol=1e-10, maxiter=50) -> BifurcationPoint

Refine a detected [`BifurcationEvent`](@ref) to a high-accuracy
[`BifurcationPoint`](@ref) using a bisection–secant hybrid.

# Arguments
- `family`       – `Vector{ContinuationPoint{T}}` from the continuation.
- `event`        – a [`BifurcationEvent`](@ref) returned by
                   [`detect_bifurcations`](@ref).
- `correct_fn`   – `(z_guess, λ) -> (z_corrected, success::Bool)`.
                   Differential corrector that converges an initial guess
                   to a periodic orbit.
- `monodromy_fn` – `(cp::ContinuationPoint) -> AbstractMatrix`.
                   Returns the 6×6 monodromy matrix for a given orbit.
- `tol`          – tolerance on `|s − s_critical|`.
- `maxiter`      – maximum number of refinement iterations.
"""
function locate_bifurcation(
    family::AbstractVector{ContinuationPoint{T}},
    event::BifurcationEvent,
    correct_fn,
    monodromy_fn;
    tol::Real  = 1e-10,
    maxiter::Int = 50,
) where {T}
    sc    = T(event.s_critical)
    get_s = event.which === :p ? first : last

    # ── Initialise the bracket ──
    cp_lo = family[event.index]
    cp_hi = family[event.index + 1]
    s_lo  = get_s(stability_index(monodromy_fn(cp_lo)))
    s_hi  = get_s(stability_index(monodromy_fn(cp_hi)))

    cp_best = abs(s_lo - sc) < abs(s_hi - sc) ? cp_lo : cp_hi
    M_best  = monodromy_fn(cp_best)

    for _ in 1:maxiter
        # Secant fraction, clamped to avoid degenerate steps
        denom = s_hi - s_lo
        frac  = abs(denom) > eps(T) ? clamp((sc - s_lo) / denom, T(0.05), T(0.95)) : T(0.5)

        λ_mid = cp_lo.λ + frac * (cp_hi.λ - cp_lo.λ)
        z_mid = cp_lo.z .+ frac .* (cp_hi.z .- cp_lo.z)

        z_corr, ok = correct_fn(z_mid, λ_mid)
        if !ok
            # Fall back to midpoint
            λ_mid = (cp_lo.λ + cp_hi.λ) / 2
            z_mid = (cp_lo.z .+ cp_hi.z) ./ 2
            z_corr, ok = correct_fn(z_mid, λ_mid)
            ok || error("correction failed during bifurcation location")
        end

        cp_mid = ContinuationPoint{T}(z_corr, λ_mid)
        M_mid  = monodromy_fn(cp_mid)
        s_mid  = get_s(stability_index(M_mid))

        # Update best
        s_best = get_s(stability_index(M_best))
        if abs(s_mid - sc) < abs(s_best - sc)
            cp_best = cp_mid
            M_best  = M_mid
        end

        abs(s_mid - sc) < tol && break

        # Narrow bracket
        if (s_mid - sc) * (s_lo - sc) < 0
            cp_hi = cp_mid;  s_hi = s_mid
        else
            cp_lo = cp_mid;  s_lo = s_mid
        end
    end

    si_best = stability_index(M_best)
    eval, evec = _find_critical_eigenpair(M_best, sc)

    return BifurcationPoint{T}(
        cp_best,
        Matrix{T}(M_best),
        si_best,
        eval,
        Vector{Complex{T}}(evec),
        event.period_mul,
    )
end

# ── Exploit (branch off) ─────────────────────────────────────────────

"""
    exploit_bifurcation(bp, correct_fn; ε=1e-4, perturbation=nothing)
        -> ContinuationPoint

Generate a first member of the new family that branches off at `bp`.

The state is perturbed by `ε` along the critical eigenvector of the
monodromy matrix (or along a user-supplied `perturbation` vector in the
free-variable space), then corrected with `correct_fn`.

# Arguments
- `bp`           – a [`BifurcationPoint`](@ref).
- `correct_fn`   – `(z_guess, λ) -> (z_corrected, success::Bool)`.
                   **Important**: for period-`N` bifurcations the corrector
                   should enforce the period `N × T` (e.g. set up a
                   shooting problem with the appropriate period).
- `ε`            – perturbation magnitude (default `1e-4`).
- `perturbation` – optional explicit perturbation vector (same length
                   as `bp.cp.z`).  When `nothing`, the real part of the
                   critical eigenvector is used.

Returns a corrected [`ContinuationPoint`](@ref) on the new family.
"""
function exploit_bifurcation(
    bp::BifurcationPoint{T},
    correct_fn;
    ε::Real = 1e-4,
    perturbation::Union{Nothing,AbstractVector} = nothing,
) where {T}
    if perturbation !== nothing
        δz = Vector{T}(perturbation)
    else
        δz = _default_perturbation(bp)
    end

    nrm = norm(δz)
    nrm > 0 || error("perturbation vector is zero — supply one explicitly via `perturbation`")
    δz ./= nrm

    z_pert = bp.cp.z .+ T(ε) .* δz
    λ      = bp.cp.λ

    z_corr, ok = correct_fn(z_pert, λ)
    ok || error("correction failed during bifurcation exploitation")

    return ContinuationPoint{T}(z_corr, λ)
end

# ── Internal helpers ──────────────────────────────────────────────────

"""
    _default_perturbation(bp) -> Vector{T}

Map the real part of the critical eigenvector (6-dimensional state space)
into the free-variable vector space.  The perturbation is placed in the
first `min(6, n)` slots of `z`; additional slots (e.g. the period) are
left at zero.
"""
function _default_perturbation(bp::BifurcationPoint{T}) where {T}
    ev = real.(bp.eigenvector)
    n  = length(bp.cp.z)
    δz = zeros(T, n)
    nstate = min(length(ev), n)
    @inbounds for i in 1:nstate
        δz[i] = ev[i]
    end
    return δz
end

"""
    _find_critical_eigenpair(M, s_critical) -> (eigenvalue, eigenvector)

Return the eigenpair of `M` whose eigenvalue is closest to the target
determined by the critical stability index.

The target eigenvalue satisfies `λ² − s λ + 1 = 0`, i.e.
`λ = (s ± √(s² − 4)) / 2`.
"""
function _find_critical_eigenpair(M::AbstractMatrix, s_critical::Real)
    disc = s_critical^2 - 4
    if disc >= 0
        target = complex((s_critical + sqrt(disc)) / 2)
    else
        target = complex(s_critical / 2, sqrt(-disc) / 2)
    end

    F    = eigen(Matrix(M))
    idx  = argmin(abs.(F.values .- target))
    return F.values[idx], F.vectors[:, idx]
end
