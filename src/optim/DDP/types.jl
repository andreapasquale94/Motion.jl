# ──────────────────────────────────────────────────────────────────────
#  Types for the constrained DDP / iLQR solver
#
#  References
#  ----------
#  [1] Lantoine & Russell, "A Hybrid Differential Dynamic Programming
#      Algorithm for Constrained Optimal Control Problems. Part 1:
#      Theory", JOTA, 154(2), 2012.
#  [2] Tassa et al., "Synthesis and Stabilization of Complex Behaviors
#      through Online Trajectory Optimization", IROS, 2012.
# ──────────────────────────────────────────────────────────────────────

# ── Constraint wrappers ──────────────────────────────────────────────

"""
    EqualityConstraint{F}

Stage equality constraint `g(x, u, t) = 0`.

# Constructor
    EqualityConstraint(g, p)

where `g(x::SVector, u::SVector, t) -> SVector{p}` returns the constraint
residual and `p` is the number of scalar constraints.
"""
struct EqualityConstraint{F}
    g::F
    p::Int
end

"""
    InequalityConstraint{F}

Stage inequality constraint `h(x, u, t) ≥ 0`.

# Constructor
    InequalityConstraint(h, q)

where `h(x::SVector, u::SVector, t) -> SVector{q}` returns the constraint
values and `q` is the number of scalar constraints.  A constraint is
*violated* when any component is negative.
"""
struct InequalityConstraint{F}
    h::F
    q::Int
end

"""
    TerminalConstraint{F}

Terminal equality constraint `ψ(xN) = 0`.

# Constructor
    TerminalConstraint(ψ, r)

where `ψ(x::SVector) -> SVector{r}` and `r` is the constraint dimension.
"""
struct TerminalConstraint{F}
    ψ::F
    r::Int
end

# ── Cost functions ───────────────────────────────────────────────────

"""
    StageCost{F}

Running cost `ℓ(x, u, t) -> scalar` evaluated at stages `k = 1, …, N-1`.
"""
struct StageCost{F}
    ℓ::F
end

"""
    TerminalCost{F}

Terminal cost `ϕ(x) -> scalar` evaluated at the final node `k = N`.
"""
struct TerminalCost{F}
    ϕ::F
end

# ── Problem definition ───────────────────────────────────────────────

"""
    DDPProblem{D, SC, TC, EC, IC, TEC}

Discrete-time constrained optimal-control problem for DDP / iLQR.

Minimises ``J = \\sum_{k=1}^{N-1} \\ell(x_k, u_k, t_k) + \\phi(x_N)``
subject to dynamics ``x_{k+1} = f(x_k, u_k, t_k, t_{k+1})``
and optional path / terminal constraints.

# Constructor
    DDPProblem(dynamics, stage_cost, terminal_cost, nx, nu;
               eq=nothing, ineq=nothing, terminal_eq=nothing)

# Arguments
- `dynamics`       – flow map `f(x, u, tₖ, tₖ₊₁) -> x_{k+1}`
- `stage_cost`     – [`StageCost`](@ref)
- `terminal_cost`  – [`TerminalCost`](@ref)
- `nx`, `nu`       – state and control dimensions
- `eq`             – [`EqualityConstraint`](@ref) or `nothing`
- `ineq`           – [`InequalityConstraint`](@ref) or `nothing`
- `terminal_eq`    – [`TerminalConstraint`](@ref) or `nothing`
"""
struct DDPProblem{D, SC, TC, EC, IC, TEC}
    dynamics::D
    stage_cost::SC
    terminal_cost::TC
    eq::EC
    ineq::IC
    terminal_eq::TEC
    nx::Int
    nu::Int
end

function DDPProblem(dynamics, stage_cost, terminal_cost, nx, nu;
                    eq=nothing, ineq=nothing, terminal_eq=nothing)
    return DDPProblem(dynamics, stage_cost, terminal_cost,
                      eq, ineq, terminal_eq, nx, nu)
end

# ── Solver options ───────────────────────────────────────────────────

"""
    DDPOptions{T}

Algorithmic parameters for the constrained DDP / iLQR solver.

# Constructor (keyword-only)
    DDPOptions(; method=:iLQR, max_iter=200, ...)

# Fields
| Field             | Default    | Description                                         |
|:------------------|:-----------|:----------------------------------------------------|
| `method`          | `:iLQR`    | `:iLQR` (Gauss-Newton) or `:DDP` (full 2nd-order)  |
| `max_iter`        | `200`      | Max inner iterations per AL outer loop               |
| `max_outer`       | `20`       | Max augmented-Lagrangian outer iterations            |
| `atol`            | `1e-8`     | Absolute cost-improvement tolerance (inner)          |
| `rtol`            | `1e-6`     | Relative cost-improvement tolerance (inner)          |
| `ctol`            | `1e-6`     | Constraint violation tolerance (outer)               |
| `μ0`              | `1.0`      | Initial AL penalty weight                            |
| `μ_max`           | `1e8`      | Maximum penalty weight                               |
| `ϕ_μ`             | `10.0`     | Penalty growth factor `μ ← ϕ_μ μ`                   |
| `reg0`            | `0.0`      | Initial regularisation on `Quu`                      |
| `reg_min`         | `1e-12`    | Minimum regularisation                               |
| `reg_max`         | `1e6`      | Maximum regularisation (failure above this)          |
| `reg_factor`      | `10.0`     | Regularisation adjustment factor                     |
| `line_search_β`   | `0.5`      | Backtracking step-size reduction                     |
| `line_search_γ`   | `1e-4`     | Armijo sufficient-decrease parameter                 |
| `verbose`         | `false`    | Print per-iteration diagnostics                      |
"""
@kwdef struct DDPOptions{T<:AbstractFloat}
    method::Symbol      = :iLQR
    max_iter::Int       = 200
    max_outer::Int      = 20
    atol::T             = 1e-8
    rtol::T             = 1e-6
    ctol::T             = 1e-6
    μ0::T               = 1.0
    μ_max::T            = 1e8
    ϕ_μ::T              = 10.0
    reg0::T             = 0.0
    reg_min::T          = 1e-12
    reg_max::T          = 1e6
    reg_factor::T       = 10.0
    line_search_β::T    = 0.5
    line_search_γ::T    = 1e-4
    verbose::Bool       = false
end

# ── Solution container ───────────────────────────────────────────────

"""
    DDPSolution{T, SX, SU}

Result returned by [`solve`](@ref).

# Fields
- `X`       – optimised state trajectory  (`Vector` of `SVector`, length `N`)
- `U`       – optimised control sequence  (`Vector` of `SVector`, length `N-1`)
- `t`       – node times                  (`Vector{T}`, length `N`)
- `J`       – final un-augmented cost
- `λ_eq`    – path equality multipliers
- `λ_ineq`  – path inequality multipliers
- `ν`       – terminal constraint multipliers
- `μ`       – final penalty weight
- `iters`   – total inner iterations
- `status`  – `:converged`, `:max_iterations`, `:regularisation_failure`, or `:line_search_failure`
"""
struct DDPSolution{T, SX, SU}
    X::Vector{SX}
    U::Vector{SU}
    t::Vector{T}
    J::T
    λ_eq::Vector{Vector{T}}
    λ_ineq::Vector{Vector{T}}
    ν::Vector{T}
    μ::T
    iters::Int
    status::Symbol
end
