module CR3BP

using StaticArrays
using LinearAlgebra
using ComponentArrays
using ForwardDiff
using NonlinearSolve

using SciMLBase: ODEProblem, ContinuousCallback, remake, solve
using SciMLBase: EnsembleProblem, EnsembleThreads

using ..Motion: Solution
import ..Motion: _compute_libration_point

include("zonal.jl")
include("base.jl")
include("stability.jl")

end
