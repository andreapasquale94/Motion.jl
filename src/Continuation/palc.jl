struct SimplePseudoArcLength{L <: AbstractResidual} <: AbstractPredictor
    model::L
end

function predict(p::SimplePseudoArcLength, hist::Vector{ContinuationPoint{T}}, ds::Real) where T 
    J = ForwardDiff.jacobian(z->residual(p.model, z, hist[end].λ), hist[end].z)
    t = vec(nullspace(J))
    t /= norm(t)
    zpred = hist[end].z + ds * t
    λpred = hist[end].λ + ds 
    return zpred, λpred, t
end