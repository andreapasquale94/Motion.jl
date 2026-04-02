
@inline function stability_index(M::AbstractMatrix)
    trM = tr(M)
    a = 2 - trM
    a2 = a * a
    b = (a2 - tr(M * M)) / 2

    disc = a2 - 4 * b + 8
    s = sqrt(disc)
    p = (a + s) / 2
    q = (a - s) / 2
    return p, q
end

function monodromy_matrix(μ::Number, x0::AbstractVector{<:Number}, t0::Number, tf::Number, alg;
	reltol = 1e-14, abstol = 1e-14, kwargs...)
	length(x0) == 6 || throw(ArgumentError("expected state of length 6, got $(length(x0))"))
	T = promote_type(typeof(μ), eltype(x0), typeof(t0), typeof(tf))
	x0v = @inbounds SVector{6, T}(x0[1], x0[2], x0[3], x0[4], x0[5], x0[6])
	p = ComponentArray(; μ = T(μ))
	xx0 = vcat(x0v, reshape(I(6), Size(36)))
	prob = ODEProblem(rhs_stm, xx0, (T(t0), T(tf)), p)
	sol = solve(prob, alg; save_everystep = false, reltol = reltol, abstol = abstol, kwargs...)
	reshape(sol.u[end][7:end], Size(6, 6))
end
