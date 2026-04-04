"""
    MDDP

Multiple-Shooting Differential Dynamic Programming module
implementing the algorithm of Pellegrini & Russell.

Decomposes the trajectory into multiple shooting legs, runs constrained
DDP on each leg independently, then corrects the shooting-node initial
conditions via a Newton step using the value-function sensitivities.

Reference: Pellegrini & Russell, "A Multiple-Shooting Differential
Dynamic Programming Algorithm", Acta Astronautica, 2020.

# Exports
- Types: `MDDPProblem`, `MDDPOptions`, `MDDPSolution`, `Leg`
- Solver: `solve`
- Re-exports from DDP: `EqualityConstraint`, `InequalityConstraint`,
  `TerminalConstraint`, `StageCost`, `TerminalCost`
"""
module MDDP

using StaticArrays
using LinearAlgebra

# Re-use DDP infrastructure
using ..DDP

export MDDPProblem, MDDPOptions, MDDPSolution, Leg
export EqualityConstraint, InequalityConstraint, TerminalConstraint
export StageCost, TerminalCost
export solve

include("types.jl")
include("leg_ddp.jl")
include("node_correction.jl")
include("solve.jl")

end # module
