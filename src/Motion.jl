module Motion

    using StaticArrays
    using LinearAlgebra 

    export CR3BPSystemProperties, BCR4BPSystemProperties
    include("models/abstract.jl")

    export rhs
    include("models/cr3bp.jl")
    include("models/bcr4bp.jl")

    export convert_state, 
           Adim, Cart, Coe, CoeRad
    include("models/state.jl")

end