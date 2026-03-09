"""
	NewtonPolynomial(nodes, values)

Newton-form interpolation polynomial backed by a divided-difference table.

- `nodes` are the interpolation abscissae.
- `values` is a matrix whose rows are the vector values at each node.

The resulting object stores both the Newton coefficients and the full divided-difference
pyramid so lower-order polynomials can be extracted without recomputing the table.
"""
struct NewtonPolynomial{T}
	nodes::Vector{T}
	coeff::Matrix{T}
	pyramid::Matrix{T}
end

function NewtonPolynomial(nodes::AbstractVector{T}, values::AbstractMatrix{T}) where {T <: Real}
	coeff, pyramid = get_newton_divdiff_coeff(nodes, values)
	return NewtonPolynomial(Vector{T}(nodes), coeff, pyramid)
end

degree(poly::NewtonPolynomial) = length(poly.nodes) - 1

dimension(poly::NewtonPolynomial) = size(poly.coeff, 2)

function (poly::NewtonPolynomial{T})(h::Real) where {T}
	return eval_newton_divdiff(T(h), poly.nodes, poly.coeff)
end

function derivative(poly::NewtonPolynomial{T}, h::Real) where {T}
	return eval_newton_divdiff_derivative(T(h), poly.nodes, poly.coeff)
end

function highest_order_coefficient(poly::NewtonPolynomial)
	return @view poly.coeff[end, :]
end

function reduced_polynomial(poly::NewtonPolynomial{T}, reduce::Integer) where {T}
	coeff = extract_newton_divdiff_coeff(poly.pyramid, degree(poly), dimension(poly), reduce)
	nodes = poly.nodes[(reduce + 1):end]
	return NewtonPolynomial(Vector{T}(nodes), coeff, zeros(T, 0, 0))
end

function get_newton_divdiff_coeff(s_vec::AbstractVector{T}, Y::AbstractMatrix{T}) where {T <: Real}
	r, c = size(Y)
	length(s_vec) == r || throw(DimensionMismatch(
		"s_vec has length $(length(s_vec)); expected $r",
	))

	pyramid = zeros(T, r, r * c)
	pyramid[:, 1:c] .= Y

	for j in 1:(r-1)
		prev_col = ((j - 1) * c + 1):(j * c)
		curr_col = (j * c + 1):((j + 1) * c)
		for i in 1:(r - j)
			ds = s_vec[i + j] - s_vec[i]
			pyramid[i, curr_col] .=
				(view(pyramid, i + 1, prev_col) .- view(pyramid, i, prev_col)) ./ ds
		end
	end

	coeff = Matrix{T}(undef, r, c)
	for i in 1:r
		coeff[i, :] .= @view pyramid[1, ((i - 1) * c + 1):(i * c)]
	end
	return coeff, pyramid
end

function nodalpoly(h::T, m::Integer, s_vec::AbstractVector{T}) where {T <: Real}
	p = one(T)
	for j in 1:m
		p *= h - s_vec[j]
	end
	return p
end

function nodalpoly_derivative(h::T, m::Integer, s_vec::AbstractVector{T}) where {T <: Real}
	m == 0 && return zero(T)

	pdiff = zero(T)
	for skip in 1:m
		term = one(T)
		for j in 1:m
			j == skip && continue
			term *= h - s_vec[j]
		end
		pdiff += term
	end
	return pdiff
end

function eval_newton_divdiff(h::T, s_vec::AbstractVector{T}, coeff::AbstractMatrix{T}) where {T <: Real}
	n, c = size(coeff)
	final_poly = zeros(T, c)
	for i in 1:n
		final_poly .+= nodalpoly(h, i - 1, s_vec) .* @view(coeff[i, :])
	end
	return final_poly
end

function eval_newton_divdiff_derivative(
	h::T,
	s_vec::AbstractVector{T},
	coeff::AbstractMatrix{T},
) where {T <: Real}
	n, c = size(coeff)
	final_poly = zeros(T, c)
	for i in 2:n
		final_poly .+= nodalpoly_derivative(h, i - 1, s_vec) .* @view(coeff[i, :])
	end
	return final_poly
end

function extract_newton_divdiff_coeff(
	pyramid::AbstractMatrix{T},
	order::Integer,
	vec_len::Integer,
	reduce::Integer = 0,
) where {T <: Real}
	npt = order + 1
	0 <= reduce < npt || throw(ArgumentError("reduce must satisfy 0 <= reduce < order + 1"))

	coeff = Matrix{T}(undef, npt - reduce, vec_len)
	for i in 1:(npt - reduce)
		coeff[i, :] .= @view pyramid[reduce + 1, ((i - 1) * vec_len + 1):(i * vec_len)]
	end
	return coeff
end

extract_nexton_divdiff_coeff(args...) = extract_newton_divdiff_coeff(args...)

function poly_error_consecutive_order(
	h::T,
	s::AbstractVector{T},
	Y::AbstractVector{T},
	C::AbstractVector{T},
	degree::Integer,
	err::T,
) where {T <: Real}
	ynorm = norm(Y)
	ynorm == 0 && return abs(err)

	nodal = nodalpoly(h, degree + 2, s)
	err_rel = abs(nodal) * norm(C) / ynorm
	return abs(err - err_rel)
end

function poly_error_consecutive_order(
	h::T,
	poly::NewtonPolynomial{T},
	Y::AbstractVector{T},
	err::T,
) where {T <: Real}
	return poly_error_consecutive_order(
		h,
		poly.nodes,
		Y,
		highest_order_coefficient(poly),
		degree(poly) - 1,
		err,
	)
end

function _consecutive_relative_error(
	h::T,
	s::AbstractVector{T},
	Y::AbstractVector{T},
	C::AbstractVector{T},
	degree::Integer,
) where {T <: Real}
	ynorm = norm(Y)
	ynorm == 0 && return zero(T)
	return abs(nodalpoly(h, degree + 2, s)) * norm(C) / ynorm
end

function solve_consecutive_step(
	h0::T,
	s::AbstractVector{T},
	Y::AbstractVector{T},
	C::AbstractVector{T},
	degree::Integer,
	target::T,
) where {T <: Real}
	target > zero(T) || return h0
	cnorm = norm(C)
	cnorm == 0 && return T(Inf)

	lower = zero(T)
	upper = max(h0, eps(T))
	f_upper = _consecutive_relative_error(upper, s, Y, C, degree)

	for _ in 1:64
		f_upper >= target && break
		upper *= T(2)
		f_upper = _consecutive_relative_error(upper, s, Y, C, degree)
	end

	f_upper < target && return upper

	for _ in 1:80
		mid = (lower + upper) / 2
		if _consecutive_relative_error(mid, s, Y, C, degree) < target
			lower = mid
		else
			upper = mid
		end
	end

	return (lower + upper) / 2
end

function solve_consecutive_step(
	h0::T,
	poly::NewtonPolynomial{T},
	Y::AbstractVector{T},
	target::T,
) where {T <: Real}
	return solve_consecutive_step(
		h0,
		poly.nodes,
		Y,
		highest_order_coefficient(poly),
		degree(poly) - 1,
		target,
	)
end
