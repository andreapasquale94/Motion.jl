module CR3BP

using StaticArrays
using LinearAlgebra
using ComponentArrays

using SciMLBase: ODEProblem, ContinuousCallback, remake, solve
using SciMLBase: EnsembleProblem, EnsembleThreads

using ..Motion: Solution

include("base.jl")

end
