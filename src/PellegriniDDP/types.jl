# ──────────────────────────────────────────────────────────────────────
#  Types for Multiple-Shooting DDP (Pellegrini & Russell)
#
#  Reference: Pellegrini & Russell, "A Multiple-Shooting Differential
#  Dynamic Programming Algorithm", Acta Astronautica, 2020.
# ──────────────────────────────────────────────────────────────────────

# Re-use constraint/cost types from DDP module
using ..DDP: EqualityConstraint, InequalityConstraint, TerminalConstraint,
             StageCost, TerminalCost

# ── Leg definition ──────────────────────────────────────────────────

"""
    Leg{T, SX, SU}

A single multiple-shooting leg containing the local trajectory,
controls, and time grid.

# Fields
- `X`  – state trajectory on this leg (Vector of SVectors, length Nₘ)
- `U`  – control sequence on this leg (Vector of SVectors, length Nₘ-1)
- `t`  – time nodes on this leg (Vector, length Nₘ)
"""
mutable struct Leg{T, SX<:SVector, SU<:SVector}
    X::Vector{SX}
    U::Vector{SU}
    t::Vector{T}
end

# ── Problem definition ──────────────────────────────────────────────

"""
    MDDPProblem{D, SC, TC, EC, IC, TEC}

Constrained optimal-control problem for multiple-shooting DDP.

Same interface as `DDPProblem` but solved via the Pellegrini & Russell
MDDP algorithm that decomposes the trajectory into multiple legs.

# Fields
- `dynamics`       – flow function `f(x, u, t_k, t_{k+1}) -> x_{k+1}`
- `stage_cost`     – running cost  `ℓ(x, u, t)`
- `terminal_cost`  – terminal cost `ϕ(x)`
- `eq`             – path equality constraints (nothing or EqualityConstraint)
- `ineq`           – path inequality constraints (nothing or InequalityConstraint)
- `terminal_eq`    – terminal equality constraints (nothing or TerminalConstraint)
- `nx::Int`        – state dimension
- `nu::Int`        – control dimension
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
    MDDPOptions

Algorithmic parameters for the multiple-shooting DDP solver.

# Fields
- `method`           – `:iLQR` or `:DDP` for the inner DDP sweeps
- `max_ddp_iter`     – max DDP iterations per leg per outer loop
- `max_outer`        – max outer (multiplier update + node correction) iterations
- `max_node_iter`    – max Newton iterations for the node correction step
- `atol`             – absolute tolerance on cost improvement (inner DDP)
- `rtol`             – relative tolerance on cost improvement (inner DDP)
- `ctol`             – constraint violation tolerance
- `dtol`             – defect (continuity) tolerance
- `μ0`               – initial augmented-Lagrangian penalty
- `μ_max`            – maximum penalty
- `ϕ_μ`              – penalty growth factor
- `reg0`             – initial DDP regularisation
- `reg_min`          – minimum regularisation
- `reg_max`          – maximum regularisation
- `reg_factor`       – regularisation growth factor
- `line_search_β`    – backtracking factor
- `line_search_γ`    – Armijo sufficient decrease parameter
- `verbose`          – print iteration info
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

Result returned by [`solve`](@ref).

# Fields
- `X`       – full optimised state trajectory  (Vector of SVectors)
- `U`       – full optimised control sequence   (Vector of SVectors)
- `t`       – full time grid                    (Vector{T})
- `J`       – final cost
- `legs`    – individual leg solutions
- `λ_eq`    – final path equality multipliers
- `λ_ineq`  – final path inequality multipliers
- `ν`       – final terminal constraint multipliers
- `μ`       – final penalty
- `iters`   – total outer iterations
- `status`  – convergence status symbol
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
