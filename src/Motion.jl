module Motion

using StaticArrays
using LinearAlgebra
using ComponentArrays

using SciMLBase: ODEProblem, ContinuousCallback, remake, solve
using SciMLBase: EnsembleProblem, EnsembleThreads
using NonlinearSolve

export Solution, SensitivitySolution, BatchSolution
include("solution.jl")

export libration_points, libration_point
include("libration_points.jl")

export compute_stretch
include("measures.jl")

# Dynamical models
include("CR3BP/CR3BP.jl")

# Algorithms
include("Continuation/Continuation.jl")
include("MultipleShooting.jl")

end
