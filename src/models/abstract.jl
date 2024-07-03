
abstract type AbstractModelProperties end

struct CR3BPSystemProperties{N <: Number} <: AbstractModelProperties
    b1::Symbol   # Name of body 1
    b2::Symbol   # Name of body 2

    # model parameters 
    μ::N         # Mass ratio

    L::N         # Length unit 
    T::N         # Time unit
    
    R1::N        # Radius of central body 
    R2::N        # Radius of smaller body 
    GM1::N       # Body 1 GM
    GM2::N       # Body 2 GM
end

struct BCR4BPSystemProperties{N <: Number} <: AbstractModelProperties
    b1::Symbol  # Name of body 1 (central body)
    b2::Symbol  # Name of body 2 (smaller body)
    b3::Symbol  # Name of body 3 (large distance body)

    # model parameters
    μ::N        # Mass ratio of the r3bp system
    μ3::N       # Normalised mass of body 3 (relative to mass of R3BP)
    a3::N       # Normalised distance between body 3 and R3BP barycenter
    ω3::N       # Angular velocity of body 3 in synodic coordinates
   
    L::N        # Length unit
    T::N        # Time unit 
    
    R1::N       # Body 1 radius 
    R2::N       # Body 2 radius 
    R3::N       # Body 3 radius 
    GM1::N      # Body 1 GM  
    GM2::N      # Body 2 GM
    GM3::N      # Body 3 GM
end
