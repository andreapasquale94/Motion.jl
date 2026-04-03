# ──────────────────────────────────────────────────────────────────────
#  Types for the constrained Differential Dynamic Programming solver
#
#  Reference: Lantoine & Russell, "A Hybrid Differential Dynamic
#  Programming Algorithm for Constrained Optimal Control Problems.
#  Part 1: Theory", JOTA 2012.
# ──────────────────────────────────────────────────────────────────────

# ── Constraint wrappers ──────────────────────────────────────────────

"""
    EqualityConstraint{F}

Wraps a callable `g(x, u, t) -> SVector{p}` representing `p` equality
constraints of the form `g(x, u, t) = 0`.
"""
struct EqualityConstraint{F}
    g::F     # g(x, u, t) -> SVector{p}
    p::Int   # number of equality constraints
end

"""
    InequalityConstraint{F}

Wraps a callable `h(x, u, t) -> SVector{q}` representing `q` inequality
constraints of the form `h(x, u, t) ≥ 0`.
"""
struct InequalityConstraint{F}
    h::F     # h(x, u, t) -> SVector{q}
    q::Int   # number of inequality constraints
end

"""
    TerminalConstraint{F}

Wraps a callable `ψ(x) -> SVector{r}` representing `r` terminal equality
constraints of the form `ψ(xN) = 0`.
"""
struct TerminalConstraint{F}
    ψ::F     # ψ(x) -> SVector{r}
    r::Int   # number of terminal constraints
end

# ── Cost functions ───────────────────────────────────────────────────

"""
    StageCost{F}

Running cost `ℓ(x, u, t) -> scalar` evaluated at each stage k = 1,…,N-1.
"""
struct StageCost{F}
    ℓ::F
end

"""
    TerminalCost{F}

Terminal cost `ϕ(x) -> scalar` evaluated at the final node.
"""
struct TerminalCost{F}
    ϕ::F
end

# ── Problem definition ───────────────────────────────────────────────

"""
    DDPProblem{D, SC, TC, EC, IC, TEC}

Full specification of a constrained optimal-control problem for DDP.

# Fields
- `dynamics::D`        – flow function `f(x, u, t_k, t_{k+1}) -> x_{k+1}`
- `stage_cost::SC`     – running cost  `ℓ(x, u, t)`
- `terminal_cost::TC`  – terminal cost `ϕ(x)`
- `eq::EC`             – path equality constraints (nothing or EqualityConstraint)
- `ineq::IC`           – path inequality constraints (nothing or InequalityConstraint)
- `terminal_eq::TEC`   – terminal equality constraints (nothing or TerminalConstraint)
- `nx::Int`            – state dimension
- `nu::Int`            – control dimension
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
    DDPOptions

Algorithmic parameters for the constrained DDP solver.

# Fields
- `max_iter`       – maximum DDP iterations per augmented-Lagrangian outer loop
- `max_outer`      – maximum augmented-Lagrangian outer iterations
- `atol`           – absolute tolerance on cost improvement for convergence
- `rtol`           – relative tolerance on cost improvement for convergence
- `ctol`           – constraint violation tolerance
- `μ0`             – initial penalty weight for augmented Lagrangian
- `μ_max`          – maximum penalty weight
- `ϕ_μ`            – penalty growth factor (μ ← ϕ_μ * μ)
- `reg0`           – initial regularisation on Quu
- `reg_min`        – minimum regularisation
- `reg_max`        – maximum regularisation
- `reg_factor`     – factor for increasing regularisation on Quu failure
- `line_search_β`  – backtracking factor
- `line_search_γ`  – sufficient decrease parameter (Armijo)
- `verbose`        – print iteration info
"""
@kwdef struct DDPOptions{T<:AbstractFloat}
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
    DDPSolution{T}

Result returned by [`solve`](@ref).

# Fields
- `X`       – optimised state trajectory  (Vector of SVectors)
- `U`       – optimised control sequence  (Vector of SVectors)
- `t`       – node times                  (Vector{T})
- `J`       – final cost
- `λ_eq`    – final Lagrange multipliers for path equality constraints
- `λ_ineq`  – final Lagrange multipliers for path inequality constraints
- `ν`       – final Lagrange multipliers for terminal constraints
- `μ`       – final penalty weight
- `iters`   – total DDP iterations (summed over outer loops)
- `status`  – convergence status symbol
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
