struct SimplePseudoArcLength{L <: AbstractResidual} <: AbstractPredictor
    model::L
end

function predict(p::SimplePseudoArcLength, hist::Vector{ContinuationPoint{T}}, ds::Real) where T 
    J = ForwardDiff.jacobian(z->residual(p.model, z, hist[end].λ), hist[end].z)
    t = vec(nullspace(J))
    t /= norm(t)

    if (length(hist) > 2)
        t_est = hist[end].z - hist[end-1].z 
        t_est /= norm(t_est)
        if ( dot(t, t_est) < 0 )
            t *= -1
        end
    end
    zpred = hist[end].z + ds * t
    λpred = hist[end].λ + ds 
    return zpred, λpred, t
end