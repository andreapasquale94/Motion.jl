abstract type AbstractShootingModel end 

struct SingleShooting{F, L <: AbstractSingleShootingLayout} <: AbstractShootingModel
    flow::F 
    layout::L
end

nx(a::SingleShooting) = nx(a.layout)
layout(s::SingleShooting) = s.layout
nvar(a::SingleShooting) = nvar(a.layout)

@inline (m::SingleShooting)(x0, T, λ) = m.flow(x0, T, λ)

# --- Residual model ---

abstract type AbstractShootingResidual <: AbstractResidual end 

struct SingleShootingResidual{S <: SingleShooting, C} <: AbstractShootingResidual
    model::S 
    constraint::C
end

function residual(sr::SingleShootingResidual, z0, λ0)
    x0, T = unpack(layout(sr.model), z0)
    x = sr.model(x0, T, λ0)
    return residual(sr.constraint, x0, x, T, λ0)
end
