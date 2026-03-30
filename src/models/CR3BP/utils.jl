
"""
	jacobi_constant(x, μ) -> T

Return the CR3BP Jacobi constant for rotating-frame state
`x = [px, py, pz, vx, vy, vz]` and mass parameter `μ`.
"""
@fastmath function jacobi_constant(x::AbstractVector{T}, μ::Number) where {T}
	μT = T(μ)
	μ1 = one(T) - μT

	@inbounds begin
		px, py, pz = x[1], x[2], x[3]
		vx, vy, vz = x[4], x[5], x[6]

		px1 = px + μT
		px2 = px - μ1

		r1 = sqrt(px1*px1 + py*py + pz*pz)
		r2 = sqrt(px2*px2 + py*py + pz*pz)

		vsq = vx*vx + vy*vy + vz*vz
		return px*px + py*py + 2*(μ1/r1 + μT/r2) - vsq
	end
end

@fastmath function jacobian(x::AbstractVector{T}, μ::Number) where T
	@inbounds px, py, pz = x[1], x[2], x[3]

	px1 = px+μ
	px2 = px-1+μ

	tmp = py*py + pz*pz
	r₁ = sqrt(px1*px1 + tmp)
	r₂ = sqrt(px2*px2 + tmp)

	r₁² = r₁*r₁
	r₂² = r₂*r₂
	r₁³ = r₁²*r₁
	r₂³ = r₂²*r₂

	f₁3 = (1-μ)/r₁³
	f₂3 = μ/r₂³
	f₁5 = f₁3/r₁²
	f₂5 = f₂3/r₂²

	tmp = f₁5 + f₂5
	uxx = 1.0 - f₁3 - f₂3 + 3*px1*px1*f₁5 + 3*px2*px2*f₂5
	uyy = 1.0 - f₁3 - f₂3 + 3*py*py*tmp
	uzz = - f₁3 - f₂3 + 3*pz*pz*tmp

	uyz = 3*py*pz*tmp
	tmp = px1*f₁5 + px2*f₂5
	uxy = 3*py*tmp
	uxz = 3*pz*tmp

	return SMatrix{6, 6, T}(
		0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
		uxx, uxy, uxz, 0.0, 2.0, 0.0,
		uxy, uyy, uyz, -2.0, 0.0, 0.0,
		uxz, uyz, uzz, 0.0, 0.0, 0.0,
	)'
end
