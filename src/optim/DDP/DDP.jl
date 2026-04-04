"""
    DDP

Constrained Differential Dynamic Programming / iterative LQR.

Implements the hybrid DDP algorithm of Lantoine & Russell [1] with
augmented-Lagrangian handling of equality and inequality constraints,
and supports both Gauss-Newton (iLQR) [2] and full second-order (DDP) modes.

# Exported types
- [`DDPProblem`](@ref) – problem specification
- [`DDPOptions`](@ref) – algorithmic parameters
- [`DDPSolution`](@ref) – optimisation result
- [`EqualityConstraint`](@ref), [`InequalityConstraint`](@ref),
  [`TerminalConstraint`](@ref) – constraint wrappers
- [`StageCost`](@ref), [`TerminalCost`](@ref) – cost wrappers

# Exported functions
- [`solve`](@ref) – run the DDP solver

# References
[1] Lantoine & Russell, "A Hybrid Differential Dynamic Programming
    Algorithm for Constrained Optimal Control Problems. Part 1: Theory",
    JOTA 154(2), 2012.

[2] Tassa et al., "Synthesis and Stabilization of Complex Behaviors
    through Online Trajectory Optimization", IROS, 2012.
"""
module DDP

using StaticArrays
using LinearAlgebra
using ForwardDiff

export DDPProblem, DDPOptions, DDPSolution
export EqualityConstraint, InequalityConstraint, TerminalConstraint
export StageCost, TerminalCost
export solve

include("types.jl")
include("derivatives.jl")
include("augmented_lagrangian.jl")
include("backward.jl")
include("forward.jl")
include("solve.jl")

end # module
