using StaticArrays

abstract type AbstractInterpolator{S} end

@inline _trasform(::Type{SVector{M, N}}, Y, n, m) where {M, N} = reshape(reinterpret(N, Y), (n, m)) 
@inline _trasform(::Type{<:AbstractVector{N}}, Y, n, m) where N = reduce(hcat, Y)

function divided_differences(x::Vector{N}, Y::Vector{<:AbstractVector{N}}) where {N <: Number}
    m = length(x)
    n = length(Y[1])
    
    # Allocate space for the pyramid 
    Δf = zeros(N, m, m*n)
    
    # Reshape dependent variables in a matrix
    @views Δf[:, 1:n] .= _trasform(typeof(Y[1]), Y, n, m)'

    for k in 1:m-1 # Loop over the columns
        for i in 1:m-k # Loop over the active rows
            δx = x[i+k] - x[i]
            # Compute divided differences 
            #                      Δf[xᵢ₊₁, ..., xᵢ₊ₖ] - Δf[xᵢ, ..., xᵢ₊ₖ₋₁]    
            # Δf[xᵢ, ..., xᵢ₊ₖ] = -------------------------------------------
            #                                     xᵢ₊ₖ - xᵢ
            @views Δf[i, k*n+1:(k+1)*n] .= ( Δf[i+1, (k-1)*n+1:k*n] .- Δf[i, (k-1)*n+1:k*n] ) ./δx
        end
    end
    coeff = reshape(view(Δf, 1, :), (n, m))
    return coeff, Δf
end

struct NewtonInterp{S, N<:Number} <: AbstractInterpolator{S}
    x::Vector{N}
    Δf::Matrix{N}
    f::Vector{N}
end

function divided_differences!(cache::NewtonInterp, x, Y)
    m = length(x)
    n = length(Y[1])
    Δf = cache.Δf

    # Reshape dependent variables in a matrix
    @views Δf[1:m, 1:n] .= _trasform(typeof(Y[1]), Y, n, m)'

    for k in 1:m-1 # Loop over the columns
        for i in 1:m-k # Loop over the active rows
            δx = x[i+k] - x[i]
            # Compute divided differences 
            #                      Δf[xᵢ₊₁, ..., xᵢ₊ₖ] - Δf[xᵢ, ..., xᵢ₊ₖ₋₁]    
            # Δf[xᵢ, ..., xᵢ₊ₖ] = -------------------------------------------
            #                                     xᵢ₊ₖ - xᵢ
            @views Δf[i, k*n+1:(k+1)*n] .= ( Δf[i+1, (k-1)*n+1:k*n] .- Δf[i, (k-1)*n+1:k*n] ) ./δx
        end
    end
    # return coefficients
    return reshape(view(Δf, 1, 1:n), (n, m))
end

function extract_divided_differences(cache::NewtonInterp, order::Int, dim::Int, reduce::Int = 0)
    n_points = order + 1
    return reshape( view(cache.Δf, 1+reduce, 1:(n_points-reduce)*dim), (n_points-reduce, dim) )
end

# def extract_nexton_divdiff_coeff(pyramid: np.ndarray, order: int, vec_len: int, reduce: int) -> np.ndarray:
#     point = order + 1
#     return np.reshape(pyramid[reduce][: (point-reduce) * vec_len], (point-reduce, vec_len))


function basis(h::N, m::Int, x::Vector{<:Number}) where {N<:Number}
    p = N(1)
    for j in 1:m
        p *= h - x[j]
    end
    return p
end

function interp(cache::NewtonInterpCache, h::Number)
    r, c = size(Δfₙ)
    m = length(cache.x)

    for i in 1:c
        @views f .+= view(Δfₙ, 1:m, i) .* basis(h, i-1, x)
    end
    return f
end


# TODO: implement most efficient version 

# using StaticArrays

# @inline _trasform(::Type{SVector{M, N}}, Y, n, m) where {M, N} = reshape(reinterpret(N, Y), (n, m)) 
# @inline _trasform(::Type{<:AbstractVector{N}}, Y, n, m) where N = reduce(hcat, Y)

# mutable struct InterpCache{T <: Number}
#     n::Int 
#     m::Int
#     x::Vector{T}
#     Δf::Matrix{T}
#     fh::Vector{T}
#     nzo::Int
# end

# function InterpCache(N::Int, x::Vector{T}, Y::Vector{<:AbstractVector{T}}) where {T<:Number}
#     # Interpolation points checks
#     m = length(x)
#     @assert length(Y) == m
#     @assert N+1 <= m  # Number of points >= poly degree - 1 

#     # Size of the interpolated vector
#     n = length(Y[1])

#     # Allocate space for the pyramid 
#     Δf0 = zeros(T, (N+1), (N+1)*n)

#     # Reshape dependent variables and insert them in the pyramid
#     # In case more points than the max order are provided, the last N 
#     # are used to build the pyramid
#     𝛶 = _trasform(typeof(Y[1]), @views(Y[end-N:end]), n, N+1)
#     @views Δf0[:, 1:n] .= 𝛶'

#     return InterpCache{T}(n, N, @views(x[end-N:end]), Δf0, zeros(T, n), N)
# end

# function InterpCache(::Type{T}, N::Int, n::Int) where {T<:Number}
#     return InterpCache{T}(n, N, zeros(T, N+1), zeros(T, N+1, (N+1)*n), zeros(T, n), N)
# end

# @inline degree(c::InterpCache) = c.m

# function divided_differences!(c::InterpCache{N}, x::Vector{N}, Y::Vector{<:AbstractVector{N}}) where {N<:Number}
#     m̄ = length(x)
#     @assert length(Y) == m̄
#     @assert c.m+1 >= m̄  # Number of points >= poly degree - 1 
#     n = c.n
#     c.nzo = m̄-1 

#     fill!(c.Δf, N(0))
#     @views c.Δf[1:m̄, 1:n] .= _trasform(typeof(Y[1]), @views(Y[end-m̄+1:end]), n, m̄)'

#     fill!(c.x, N(0))
#     @views c.x[1:m̄] .= x 

#     # Loop over the columns 
#     for k in 1:m̄-1 
#         # Loop over the active rows
#         for i in 1:m̄-k
#             δx = x[i+k] - x[i]
#             @views c.Δf[i, k*n+1:(k+1)*n] .= (c.Δf[i+1, (k-1)*n+1:k*n] .- c.Δf[i, (k-1)*n+1:k*n]) ./ δx
#         end
#     end
#     nothing
# end

# function basis(c::InterpCache{<:Number}, m::Int, h::N) where {N<:Number}
#     p = N(1)
#     for j in 1:m
#         p *= h - c.x[j]
#     end
#     return p
# end

# function interp(c::InterpCache{<:Number}, h::N)
#     Δfₙ = reshape(view(c.Δf, 1, :), (c.n, c.nzo+1)) 
#     for i in 1:c.nzo+1
#         @views c.fh .+= view(Δfₙ, :, i) .* basis(h, i-1, c.x)
#     end
#     return c.fh
# end
