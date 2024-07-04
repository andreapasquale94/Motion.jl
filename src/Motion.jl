module Motion

    using StaticArrays
    using LinearAlgebra 

    include("utils/root.jl")

    export CR3BPSystemProperties, BCR4BPSystemProperties
    include("models/abstract.jl")

    export translate, rotate, transform, 
           Adim,  # synodic 
           AdimCart, Cart, Coe, CoeRad # inertial
    include("models/state.jl")

    export rhs3b, rhs_stm3b, libration_points
    include("models/cr3bp.jl")
    include("models/bcr4bp.jl")
    include("models/lagrange.jl")

    export EM3B_PROPERTIES, SE3B_PROPERTIES
    include("models/predefined.jl")

end