# ──────────────────────────────────────────────────────────────────────
#  Types for Multiple-Shooting DDP (Pellegrini & Russell)
#
#  Reference: Pellegrini & Russell, "A Multiple-Shooting Differential
#  Dynamic Programming Algorithm. Part 1: Theory",
#  Acta Astronautica 170, 2020.
# ──────────────────────────────────────────────────────────────────────

# Re-use constraint/cost types from DDP module
using ..DDP: EqualityConstraint, InequalityConstraint, TerminalConstraint,
             StageCost, TerminalCost

# ── Leg definition ──────────────────────────────────────────────────

"""
    Leg{T, SX, SU}

A single multiple-shooting leg containing the local state trajectory,
controls, and time grid.  Each leg is an independent sub-problem
whose initial state is a decision variable updated by the outer
node-correction step.

# Fields
- `X`  – state trajectory `[x₁, …, x_{Nₘ}]`  (`Vector{SX}`)
- `U`  – control sequence  `[u₁, …, u_{Nₘ-1}]` (`Vector{SU}`)
- `t`  – time nodes        `[t₁, …, t_{Nₘ}]`   (`Vector{T}`)
"""
mutable struct Leg{T, SX<:SVector, SU<:SVector}
    X::Vector{SX}
    U::Vector{SU}
    t::Vector{T}
end

# ── Problem definition ──────────────────────────────────────────────

"""
    MDDPProblem{D, SC, TC, EC, IC, TEC}

Constrained optimal-control problem for the multiple-shooting DDP
algorithm of Pellegrini & Russell (2020).

Identical interface to [`DDPProblem`](@ref) but solved via the MDDP
three-loop architecture:

1. **Outer loop** – augmented-Lagrangian multiplier / penalty updates
2. **Node correction** – Newton step on shooting-node initial states
   using value-function sensitivities and state-transition matrices
3. **Inner loop** – DDP / iLQR sweeps on each leg independently

# Constructor
    MDDPProblem(dynamics, stage_cost, terminal_cost, nx, nu;
                eq=nothing, ineq=nothing, terminal_eq=nothing)

# Arguments
- `dynamics`       – flow map `f(x, u, tₖ, tₖ₊₁) -> xₖ₊₁`
- `stage_cost`     – [`StageCost`](@ref) wrapper
- `terminal_cost`  – [`TerminalCost`](@ref) wrapper
- `nx`, `nu`       – state and control dimensions
- `eq`             – [`EqualityConstraint`](@ref) or `nothing`
- `ineq`           – [`InequalityConstraint`](@ref) or `nothing`
- `terminal_eq`    – [`TerminalConstraint`](@ref) or `nothing`
"""
struct MDDPProblem{D, SC, TC, EC, IC, TEC}
    dynamics::D
    stage_cost::SC
    terminal_cost::TC
    eq::EC
    ineq::IC
    terminal_eq::TEC
    nx::Int
    nu::Int
end

function MDDPProblem(dynamics, stage_cost, terminal_cost, nx, nu;
                     eq=nothing, ineq=nothing, terminal_eq=nothing)
    return MDDPProblem(dynamics, stage_cost, terminal_cost,
                       eq, ineq, terminal_eq, nx, nu)
end

# ── Solver options ──────────────────────────────────────────────────

"""
    MDDPOptions{T}

Algorithmic parameters for the multiple-shooting DDP solver.

# Constructor (keyword-only)
    MDDPOptions(; method=:iLQR, max_ddp_iter=100, ...)

# Fields
| Field             | Default    | Description                                           |
|:------------------|:-----------|:------------------------------------------------------|
| `method`          | `:iLQR`    | `:iLQR` (Gauss-Newton) or `:DDP` (full 2nd-order)    |
| `max_ddp_iter`    | `100`      | Max DDP iterations per leg per outer loop             |
| `max_outer`       | `30`       | Max augmented-Lagrangian outer iterations             |
| `max_node_iter`   | `20`       | Max Newton iterations for node correction             |
| `atol`            | `1e-8`     | Absolute cost-improvement tolerance (inner DDP)       |
| `rtol`            | `1e-6`     | Relative cost-improvement tolerance (inner DDP)       |
| `ctol`            | `1e-6`     | Constraint violation tolerance (outer)                |
| `dtol`            | `1e-6`     | Defect (continuity) tolerance                         |
| `μ0`              | `1.0`      | Initial AL penalty weight                             |
| `μ_max`           | `1e8`      | Maximum penalty weight                                |
| `ϕ_μ`             | `10.0`     | Penalty growth factor `μ ← ϕ_μ μ`                    |
| `reg0`            | `0.0`      | Initial DDP regularisation on `Quu`                   |
| `reg_min`         | `1e-12`    | Minimum regularisation                                |
| `reg_max`         | `1e6`      | Maximum regularisation (failure above this)           |
| `reg_factor`      | `10.0`     | Regularisation adjustment factor                      |
| `line_search_β`   | `0.5`      | Backtracking step-size reduction factor               |
| `line_search_γ`   | `1e-4`     | Armijo sufficient-decrease parameter                  |
| `verbose`         | `false`    | Print per-iteration diagnostics                       |
"""
@kwdef struct MDDPOptions{T<:AbstractFloat}
    method::Symbol      = :iLQR
    max_ddp_iter::Int   = 100
    max_outer::Int      = 30
    max_node_iter::Int  = 20
    atol::T             = 1e-8
    rtol::T             = 1e-6
    ctol::T             = 1e-6
    dtol::T             = 1e-6
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

# ── Solution container ──────────────────────────────────────────────

"""
    MDDPSolution{T, SX, SU}

Result returned by [`solve`](@ref) for the MDDP solver.

# Fields
- `X`       – full optimised state trajectory  (`Vector{SX}`)
- `U`       – full optimised control sequence  (`Vector{SU}`)
- `t`       – full time grid                   (`Vector{T}`)
- `J`       – final un-augmented cost
- `legs`    – per-leg solutions (`Vector{Leg}`)
- `λ_eq`    – flattened path equality multipliers
- `λ_ineq`  – flattened path inequality multipliers
- `ν`       – terminal constraint multipliers
- `μ`       – final penalty weight
- `iters`   – total outer iterations
- `status`  – `:converged` or `:max_iterations`
"""
struct MDDPSolution{T, SX, SU}
    X::Vector{SX}
    U::Vector{SU}
    t::Vector{T}
    J::T
    legs::Vector{Leg{T, SX, SU}}
    λ_eq::Vector{Vector{T}}
    λ_ineq::Vector{Vector{T}}
    ν::Vector{T}
    μ::T
    iters::Int
    status::Symbol
end
