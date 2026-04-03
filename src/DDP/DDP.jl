"""
    DDP

Constrained Differential Dynamic Programming (DDP) module implementing the
hybrid DDP algorithm with augmented-Lagrangian handling of equality and
inequality constraints.

Reference: Lantoine & Russell, "A Hybrid Differential Dynamic Programming
Algorithm for Constrained Optimal Control Problems. Part 1: Theory",
Journal of Optimization Theory and Applications, 2012.

# Exports
- Types: `DDPProblem`, `DDPOptions`, `DDPSolution`,
  `EqualityConstraint`, `InequalityConstraint`, `TerminalConstraint`,
  `StageCost`, `TerminalCost`
- Solver: `solve`
"""
module DDP

using StaticArrays
using LinearAlgebra

export DDPProblem, DDPOptions, DDPSolution
export EqualityConstraint, InequalityConstraint, TerminalConstraint
export StageCost, TerminalCost
export solve

include("types.jl")
include("derivatives.jl")
include("backward.jl")
include("forward.jl")
include("solve.jl")

end # module
