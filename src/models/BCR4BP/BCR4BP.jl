module BCR4BP

using StaticArrays
using LinearAlgebra
using ComponentArrays

using SciMLBase: ODEProblem, ContinuousCallback, remake, solve
using SciMLBase: EnsembleProblem, EnsembleThreads

using ..Motion: Solution, __acceleration_to_cartesian

include("base.jl")

export rhs_spm, rhs_pms, make, flow, build_solution

end
