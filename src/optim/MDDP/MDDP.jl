"""
    MDDP

Multiple-Shooting Differential Dynamic Programming.

Decomposes the trajectory into `M` shooting legs, runs constrained DDP
on each leg independently, then corrects the shooting-node initial
conditions via a Newton step using value-function sensitivities.  This
improves robustness for highly sensitive dynamics (e.g. multi-revolution
low-thrust in the CR3BP) compared to single-shooting DDP.

# Exported types
- [`MDDPProblem`](@ref), [`MDDPOptions`](@ref), [`MDDPSolution`](@ref)
- [`Leg`](@ref)
- Re-exports from DDP: `EqualityConstraint`, `InequalityConstraint`,
  `TerminalConstraint`, `StageCost`, `TerminalCost`

# Exported functions
- [`solve`](@ref)

# References
Pellegrini & Russell, "A Multiple-Shooting Differential Dynamic
Programming Algorithm. Part 1: Theory", Acta Astronautica 170, 2020.
"""
module MDDP

using StaticArrays
using LinearAlgebra

# Re-use the full DDP infrastructure
using ..DDP
using ..DDP: cost_derivatives, terminal_cost_derivatives,
             dynamics_derivatives, dynamics_hessians,
             constraint_derivatives, terminal_constraint_derivatives,
             augment_Q_equality, augment_Q_inequality,
             eval_stage_al_cost, eval_terminal_al_cost,
             init_path_multipliers, init_terminal_multipliers,
             update_multipliers!,
             _terminal_init, _Q_expansion

export MDDPProblem, MDDPOptions, MDDPSolution, Leg
export EqualityConstraint, InequalityConstraint, TerminalConstraint
export StageCost, TerminalCost
export solve

include("types.jl")
include("leg_ddp.jl")
include("node_correction.jl")
include("solve.jl")

end # module
