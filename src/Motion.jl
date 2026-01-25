module Motion

using StaticArrays
using LinearAlgebra
using ComponentArrays

using SciMLBase: ODEProblem, ContinuousCallback, remake, solve
using SciMLBase: EnsembleProblem, EnsembleThreads

include("root.jl")

export libration_points
include("model/libration_points.jl")

export Solution, SensitivitySolution, BatchSolution
include("model/solution.jl")

export compute_stretch
include("model/measures.jl")

include("model/utils.jl")

export make_cr3bp, flow_cr3bp, solve_cr3bp
include("model/cr3bp/utils.jl")
include("model/cr3bp/base.jl")
include("model/cr3bp/batch.jl")

export make_cr3bp_constant_thrust, flow_cr3bp_constant_thrust, solve_cr3bp_constant_thrust
include("model/cr3bp/constant_thrust.jl")

# Optimisation modules
include("opt/ImpulsiveShooting.jl")

end
