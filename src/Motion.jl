module Motion

using StaticArrays
using LinearAlgebra
using ComponentArrays

using SciMLBase: ODEProblem, ContinuousCallback, remake, solve
using SciMLBase: EnsembleProblem, EnsembleThreads

include("root.jl")

export Solution, SensitivitySolution, BatchSolution
include("solution.jl")

export libration_points
include("models/libration_points.jl")

export compute_stretch
include("models/measures.jl")
include("models/utils.jl")

# Models
include("models/CR3BP/CR3BP.jl")
include("models/ER3BP/ER3BP.jl")
include("models/BCR4BP/BCR4BP.jl")

# Optimisation modules
include("opt/ImpulsiveShooting.jl")
include("opt/ConstThrustShooting.jl")

# Continuation
include("Continuation/Continuation.jl")

end
