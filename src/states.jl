"""
General state types
"""

abstract type AbstractState{T} end
abstract type AbstractAbsoluteState{T} <: AbstractState{T} end
abstract type AbstractUncertainState{T} <: AbstractState{T} end

const Covariance{T} = Array{T,2} where T


"""
    DiscreteState(x::Vector[, t::Int64])
    DiscreteState(x::AbstractFloat[, t::Int64])

Discrete-time state. Contains a state vector and an Integer-valued time step. If
not provided, the default value of `t` is 0.
"""
mutable struct DiscreteState{T<:AbstractFloat} <: AbstractAbsoluteState{T}
    x::Vector{T}
    t::Int64

    DiscreteState(x::Vector{T}, t::Int64) where {T} = new{T}(x, t)
    DiscreteState(x::Vector{T}) where {T} = new{T}(x, 0)
end
DiscreteState(x::AbstractFloat) = DiscreteState([x])
DiscreteState(x::AbstractFloat, t::Int64) = DiscreteState([x], t)

"""
    UncertainDiscreteState(x::Vector, P::Matrix[, t::Int64])
    UncertainDiscreteState(x::AbstractFloat, P::AbstractFloat[, t::Int64])

Uncertain discrete-time state. Contains a state vector, a covariance matrix
representing the uncertainty in that state, and an Integer-valued time step. If
not provided, the default value of `t` is 0.

Uncertain states are Gaussian distributioned random varaibles with mean \$x\$
and covariance \$P\$.
"""
mutable struct UncertainDiscreteState{T<:AbstractFloat} <:
    AbstractUncertainState{T}

    x::Vector{T}
    P::Covariance{T}
    t::Int64

    function UncertainDiscreteState(x::Vector{T}, P::Covariance{T}, t) where T
            if length(x) != size(P)[1] || length(x) != size(P)[2]
                throw(DimensionMismatch(
                    "Dimensions of covariance and state do not match."))
            end
            new{T}(x, P, t)
    end
    function UncertainDiscreteState(x::Vector{T}, P::Covariance{T}) where T
            if length(x) != size(P)[1] || length(x) != size(P)[2]
                throw(DimensionMismatch(
                    "Dimensions of covariance and state do not match."))
            end
            new{T}(x, P, 0)
    end
end
UncertainDiscreteState{T<:AbstractFloat}(x::T, P::T) =
    UncertainDiscreteState([x], reshape([P],1,1))
UncertainDiscreteState{T<:AbstractFloat}(x::T, P::T, t) =
    UncertainDiscreteState([x], reshape([P],1,1), t)

"""
Union of Discrete state types (absolute and uncertain).
"""
const UnionDiscrete{T} = Union{DiscreteState{T}, UncertainDiscreteState{T}} where T


"""
    ContinuousState(x::Vector[, t::AbstractFloat])
    ContinuousState(x::AbstractFloat[, t::AbstractFloat])

Continuous-time state. Contains a state vector and a floating-point time step.
If not provided, the default value of `t` is 0.0.
"""
mutable struct ContinuousState{T<:AbstractFloat} <: AbstractAbsoluteState{T}
    x::Vector{T}
    t::T

    ContinuousState(x::Vector{T}, t::T) where {T} = new{T}(x, t)
    ContinuousState(x::Vector{T}) where {T} = new{T}(x, 0.0)
end
ContinuousState(x::AbstractFloat) = ContinuousState([x])
ContinuousState{T<:AbstractFloat}(x::T, t::T) = ContinuousState([x], t)

"""
    UncertainContinuousState(x::Vector, P::Matrix[, t::AbstractFloat])
    UncertainContinuousState(x::AbstractFloat, P::AbstractFloat[, t::AbstractFloat])

Uncertain continuous-time state. Contains a state vector, a covariance matrix
representing the uncertainty in that state, and a floating-point time step. If
not provided, the default value for `t` is 0.0.

Uncertain states are Gaussian distributioned random varaibles with mean \$x\$
and covariance \$P\$.
"""
mutable struct UncertainContinuousState{T<:AbstractFloat} <:
    AbstractUncertainState{T}

    x::Vector{T}
    P::Covariance{T}
    t::T

    function UncertainContinuousState(x::Vector{T}, P::Covariance{T},
        t::T) where T
            if length(x) != size(P)[1] || length(x) != size(P)[2]
                throw(DimensionMismatch(
                    "Dimensions of covariance and state do not match."))
            end
            new{T}(x, P, t)
    end
    function UncertainContinuousState(x::Vector{T}, P::Covariance{T}) where T
            if length(x) != size(P)[1] || length(x) != size(P)[2]
                throw(DimensionMismatch(
                    "Dimensions of covariance and state do not match."))
            end
            new{T}(x, P, 0.0)
    end
end
UncertainContinuousState{T<:AbstractFloat}(x::T, P::T) =
    UncertainContinuousState([x], reshape([P],1,1))
UncertainContinuousState{T<:AbstractFloat}(x::T, P::T, t::T) =
    UncertainContinuousState([x], reshape([P],1,1), t)
    
"""
Union of Continuous state types (absolute and uncertain).
"""
const UnionContinuous{T} = Union{ContinuousState{T}, UncertainContinuousState{T}} where T


"""
    make_uncertain(state::AbstractUncertainState[, P::Matrix])

Create an uncertain copy of the input absolute state. If not provided, the
covariance matrix, `P`, is all zero.
"""
make_uncertain{T}(state::DiscreteState{T}) =
    UncertainDiscreteState(
        state.x, zeros(T, length(state.x), length(state.x)), state.t)
make_uncertain{T}(state::DiscreteState{T}, P::Covariance{T}) =
    UncertainDiscreteState(state.x, P, state.t)
make_uncertain{T}(state::ContinuousState{T}) =
    UncertainContinuousState(
        state.x, zeros(T, length(state.x), length(state.x)), state.t)
make_uncertain{T}(state::ContinuousState{T}, P::Covariance{T}) =
    UncertainContinuousState(state.x, P, state.t)

"""
    make_absolute(state::AbstractUncertainState)

Create an absolute copy of the input uncertain state.
"""
make_absolute(state::UncertainDiscreteState) = DiscreteState(state.x, state.t)
make_absolute(state::UncertainContinuousState) =
    ContinuousState(state.x, state.t)


"""
"""
absolute_type{T}(state::UncertainDiscreteState{T}) = DiscreteState{T}
absolute_type{T}(state::UncertainContinuousState{T}) = ContinuousState{T}

"""
"""
uncertain_type{T}(state::DiscreteState{T}) = UncertainDiscreteState{T}
uncertain_type{T}(state::ContinuousState{T}) = UncertainContinuousState{T}


"""
    sample(state::AbstractUncertainState)

Sample uncertain state. Returns absolute state of the corresponding time
format (i.e. UncertainDiscreteState => DiscreteState).
"""
function sample{T}(state::UncertainDiscreteState{T})
    if all(state.P .== 0)
        sampled_state = state.x
    else
        sampled_state = state.x+chol(Hermitian(state.P))*randn(T, length(state.x))
    end
    return DiscreteState(sampled_state, state.t)
end
function sample{T}(state::UncertainContinuousState{T})
    if all(state.P .== 0)
        sampled_state = state.x
    else
        sampled_state = state.x+chol(Hermitian(state.P))*randn(T, length(state.x))
    end
    return ContinuousState(sampled_state, state.t)
end


"""
    distance(state1::AbstractState, state2::AbstractState)

Computes the Euclidean distance between two states. Ignores uncertainty in
either of the states.

\$d(x,y) = \\left\\|{x-y}\\right\\|\$
"""
function distance{T}(state1::AbstractState{T}, state2::AbstractState{T})
    return sqrt(sum((state1.x .- state2.x).^2))
end


"""
    mahalanobis(x::AbstractAbsoluteState, y::AbstractUncertainState)
    mahalanobis(x::AbstractAbsoluteState, y::AbstractAbsoluteState, P::Matrix)

Compute the Mahalanobis distance between a point and a distribution or two
points in the same distribution. If `P` is the covariance of the distribution,

\$d(x,y) = \\sqrt{(x-y)^T P^{-1} (x-y)}\$
"""
function mahalanobis{T}(x::AbstractAbsoluteState{T},
                        y::AbstractUncertainState{T})
    diff_vec = x.x .- y.x
    return sqrt(diff_vec'*inv(y.P)*diff_vec)
end
function mahalanobis{T}(x::AbstractAbsoluteState{T},
                        y::AbstractAbsoluteState{T}, P::Array{T,2})
    diff_vec = x.x .- y.x
    return sqrt(diff_vec'*inv(P)*diff_vec)
end



"""
State addition.
"""
function Base.:+(state::AbstractState, a)
    out = deepcopy(state)
    out.x .+= a
    return out
end
function Base.:+(a, state::AbstractState)
    out = deepcopy(state)
    out.x .+= a
    return out
end
function Base.broadcast!(::typeof(+), out::AbstractState, state::AbstractState,
                         a)
    out.x .= state.x .+ a
    return nothing
end
function Base.broadcast!(::typeof(+), out::AbstractState, a,
                         state::AbstractState)
    out.x .= state.x .+ a
    return nothing
end


"""
State subtraction.
"""
function Base.:-(state::AbstractState, a)
    out = deepcopy(state)
    out.x .-= a
    return out
end
function Base.:-(a, state::AbstractState)
    out = deepcopy(state)
    out.x .-= a
    return out
end
function Base.broadcast!(::typeof(-), out::AbstractState, state::AbstractState,
                         a)
    out.x .= state.x .- a
    return nothing
end
function Base.broadcast!(::typeof(-), out::AbstractState, a,
                         state::AbstractState)
    out.x .= state.x .- a
    return nothing
end


"""
State equality check.
"""
function Base.:(==){S<:AbstractAbsoluteState}(state1::S, state2::S)
    if state1.x == state2.x && state1.t == state2.t
        return true
    else
        return false
    end
end
function Base.:(==){S<:AbstractUncertainState}(state1::S, state2::S)
    if state1.x == state2.x && state1.P == state2.P && state1.t == state2.t
        return true
    else
        return false
    end
end
Base.:(==)(state1::AbstractState, state2::AbstractState) = false


"""
State multiplication.
"""
function Base.:*(state::AbstractState, a::Number)
    out = deepcopy(state)
    out .*= a
    return out
end
function Base.:*(a::Number, state::AbstractState)
    out = deepcopy(state)
    out .*= a
    return out
end
function Base.broadcast!(::typeof(*), out::AbstractAbsoluteState,
                         state::AbstractState, a::Number)
    out.x .= state.x .* a
    return nothing
end
function Base.broadcast!(::typeof(*), out::AbstractAbsoluteState, a::Number,
                         state::AbstractState)
    out.x .= state.x .* a
    return nothing
end
function Base.broadcast!(::typeof(*), out::AbstractUncertainState, a::Number,
                         state::AbstractUncertainState)
    out.x .= state.x .* a
    out.P .= state.P .* a^2
    return nothing
end
function Base.broadcast!(::typeof(*), out::AbstractUncertainState,
                         state::AbstractUncertainState, a::Number)
    out.x .= state.x .* a
    out.P .= state.P .* a^2
    return nothing
end

function Base.:*(A::Matrix, state::AbstractState)
    out = deepcopy(state)
    out .= A .* out
    return out
end
function Base.broadcast!(::typeof(*), out::AbstractAbsoluteState, A::Matrix,
                         state::AbstractState)
    out.x .= A * state.x
    return nothing
end
function Base.broadcast!(::typeof(*), out::AbstractUncertainState, A::Matrix,
                         state::AbstractUncertainState)
    out.x .= A * state.x
    out.P .= A * state.P * A'
    return nothing
end


"""
State assignment operations.
"""
function Base.broadcast!(::typeof(identity), out::DiscreteState,
                            state::DiscreteState)
    out.x .= state.x
    out.t = state.t
    return nothing
end
function Base.broadcast!(::typeof(identity), out::ContinuousState,
                            state::ContinuousState)
    out.x .= state.x
    out.t = state.t
    return nothing
end
function Base.broadcast!(::typeof(identity), out::UncertainDiscreteState,
                            state::UncertainDiscreteState)
    out.x .= state.x
    out.P .= state.P
    out.t = state.t
    return nothing
end
function Base.broadcast!(::typeof(identity), out::UncertainContinuousState,
                            state::UncertainContinuousState)
    out.x .= state.x
    out.P .= state.P
    out.t = state.t
    return nothing
end
