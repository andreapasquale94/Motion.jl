
using LinearAlgebra
using StaticArrays
using SciMLBase

# Multiple shooting handling either ballistic or manoeuvered arcs

struct ArcCache{S, N}
    tspan::MVector{2, N}            # Time span of the arc [t₀, t₁]
    u0::MVector{S, N}               # Initial state u(t₀)
    u1::MVector{S, N}               # Final state u(t₁)
    par::Vector{N}                  # Arc manoeuvre parameters [dv, ras, dec]
    du0_dt::MVector{S, N}           # Partial of u(t₀) w.r.t. initial time, t₀
    du1_dT::MVector{S, N}           # Partial of u(t₁) w.r.t. arc duration, T = t₁-t₀
    du1_du0::MMatrix{S, S, N}       # Partial of u(t₁) w.r.t. u(t₀)
    du1_dpar::Matrix{N}             # Partial of u(t₁) w.r.t. p 
end

function ArcCache{S, N}(; tspan=missing, u0=missing, par=missing) where {S, N}
    if tspan === missing
        tspan = zeros(N, 2)
    else
        @assert length(tspan) == 2
    end
    if u0 === missing
        u0 = zeros(N, S)
    else
        @assert length(u0) == N
    end
    u1 = zeros(N, S)
    if par === missing
        par = zeros(N, 0)
    end
    du0_dt = zeros(N, S)
    du1_dT = zeros(N, S)
    du1_du0 = zeros(N, S, S)
    du1_dp = zeros(N, S, length(par))
    return ArcCache{S, N}(tspan, u0, u1, par, du0_dt, du1_dT, du1_du0, du1_dp)
end

function update!(c::ArcCache{S, N}; tspan=missing, u0=missing, u1=missing, par=missing, 
    du0_dt=missing, du1_dT=missing, du1_du0=missing, du1_dp=missing
) where {S, N}
    !(tspan === missing) && c.tspan .= tspan
    !(u0 === missing) && c.u0 .= u0
    !(u1 === missing) && c.u1 .= u1
    !(par === missing) && c.par .= par
    !(du0_dt === missing) && c.du0_dt .= du0_dt
    !(du1_dT === missing) && c.du1_dT .= du1_dT
    !(du1_du0 === missing) && c.du1_du0 .= du1_du0
    !(du1_dp === missing) && c.du1_dp .= tspan
    nothing
end

@inline hasparam(c::ArcCache{<:Any, N}) where N = length(c.par) > zero(N)
@inline duration(c::ArcCache{<:Any, N}) where N = abs(c.tspan[2] - c.tspan[1])
@inline isforward(c::ArcCache{<:Any, N}) where N = c.tspan[2] - c.tspan[1] ≥ zero(N)

# ==========================================================================================

struct SubSegmentData{S, N}
    tstops::Vector{N}
    arcs::Vector{ArcCache{S, N}}
end

@inline hasparam(s::SubSegmentData, i::Int) = hasparam(s.arcs[i])
@inline duration(s::SubSegmentData, i::Int) = duration(s.arcs[i])
@inline isforward(s::SubSegmentData, i::Int) = isforward(s.arcs[i])

@inline Base.length(s::SubSegmentData) = length(s.arcs)

@inline function duration(s::SubSegmentData{<:Any, N}) where N
    @inbounds if length(s.tstops) > 0
        return abs(s.tstops[end] - s.tstops[1])
    end
    return zero(N)
end

@inline function isforward(s::SubSegmentData{<:Any, N}) where N
    @inbounds if length(s.tstops) > 0
        return s.tstops[end] - s.tstops[1] ≥ zero(N)
    end
    return false
end

@inline function hasparam(s::SubSegmentData) where N 
    flag = false
    @inbounds if length(s) > 0
        for i in 1:length(s)
            flag = flag || hasparam(s, i)
        end
        return flag
    end
    return false
end

function SubSegmentData{S, N}(; tstops=missing, u0s=missing, pars=missing) where {S, N}     
    n_stops = length(tstops) 
    if tstops === missing || n_stops == 0 
        return SubSegmentData{S, N}(N[], ArcCache{S, N}[])
    end
    
    n_arcs = length(tstops) - 1
    @assert n_arcs > 0 

    has_u0 = u0s !== missing
    has_par = pars !== missing && length(pars) != 0

    has_u0 && @assert length(u0s) ∈ ( n_stops, n_arcs )
    has_par && @assert length(pars) == n_arcs

    # Initialize arcs vector
    arcs = Vector{ArcCache{S, N}}(undef, n_arcs)

    @inbounds for i in 1:n_arcs
        arcs[i] = ArcCache{S, N}(
            tspan = @views(tstops[i:i+1]),
            u0 = has_u0 ? u0s[i] : missing, par = has_par ? pars[i] : missing
        )
    end
    return SubSegmentData{S, N}(tstops, arcs)
end

# ==========================================================================================
 
mutable struct SubSegment{F, DF, C, D}
    f::F
    df::DF
    cb::C
    const data::D
end

function Base.show(io::IO, mime::MIME"text/plain", s::SubSegment)
    if isforward(s.data)
        printstyled(io, "Forward"; bold=true)
    else 
        printstyled(io, "Backward"; bold=true)
    end 
    printstyled(io, " SubSegment"; color=:cyan)
    print(io, " with ")
    printstyled(io, "$(length(s.data))"; color=:cyan)
    print(io, " arcs. Manoeuvres: ")
    printstyled(io, "$(hasparam(s.data))"; color=:cyan)
    println(io, "")
    
    printstyled(io, "Duration: ", italic=true)
    print(io, "$(duration(s.data))")
end

@inline __subsegment_callback_condition(data::SubSegmentData, u, t, int) = t ∈ data.tstops 

function __subsegment_callback_affect(data::SubSegmentData{S, N}, int) where {S, N}
    j = searchsortedlast( t->int.t == t, data.tstops ) # find index of the arc
    arc = data.arcs[j]

    @inbounds if hasparam(arc)
        # if the arc has a manoeuvre, update the state (assume manouvre parametrization)
        dv, ras, dec = arc.par     
        sα, cα = sincos(ras)
        sβ, cβ = sincos(dec)

        int.u[4] += dv*cβ*cα
        int.u[5] += dv*cβ*sα
        int.u[6] += dv*sβ
    end
    nothing
end

function SubSegment(
    fun::SciMLBase.AbstractODEFunction, dfun::SciMLBase.AbstractODEFunction, 
    u0::AbstractVector{N}, tstops, p; 
    u0s=missing, pars=missing) where {N}
    S = length(u0)

    # Create data first 
    data = SubSegmentData{S, N}(tstops=tstops, u0s=u0s, pars=pars)
    
    # Create callback for the points along the trajectory
    cb = DiscreteCallback(
        (u, t, int) -> __subsegment_callback_condition(data, u, t, int),
        int -> __subsegment_callback_affect(data, int),
        save_positions=(true, true)
    )

    if length(tstops) == 0
        tspan = [zero(N), zero(N)]
    else
        tspan = (tstops[1], tstops[end])
    end
    f = ODEProblem(fun, u0, tspan, p)
    u00 = vcat(u0, reshape(I(S), Size(S*S,)))
    df = ODEProblem(dfun, u00, tspan)

    return SubSegment{typeof(f), typeof(df), typeof(cb), SubSegmentData{S, N}}(
        f, df, cb, data
    )
end

# ==========================================================================================

struct Segment{S}
    forward::S
    backward::S
end

 function Segment(fun, dfun, u0, fwstops, bwstops, p)
    fw = SubSegment(fun, dfun, u0, fwstops, p)
    bw = SubSegment(fun, dfun, u0, bwstops, p)

    T = promote_type(typeof(fw), typeof(bw))
    return Segment{T}(fw, bw)
end


u0 = SA[1.0000386056378008, -1.5157926653278972e-5, 0.0, 0.12631964272920515, 0.3470603660320666, 0.0]
fun = ODEFunction(rhs3b)
dfun = ODEFunction(rhs_stm3b)
steps = [0.0, 0.04047280509064317, 1.2913291661796644]

s = Segment(fun, dfun, u0, steps, [], Motion.SE3B_PROPERTIES)


function Base.show(io::IO, mime::MIME"text/plain", s::Segment)
    printstyled(io, "Segment"; color=:cyan)
    print(io, " with ")
    printstyled(io, "$(length(s.forward.data))"; color=:cyan)
    print(io, " forward and ")
    printstyled(io, "$(length(s.backward.data))"; color=:cyan)
    print(io, " backward arcs")
    println(io, "")
    printstyled(io, "State: ", italic=true)
    show(io, mime, s.forward.f.u0)
end
