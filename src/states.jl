"""
General state types
"""

abstract type AbstractState{T} end
abstract type AbstractAbsoluteState{T} <: AbstractState{T} end
abstract type AbstractUncertainState{T} <: AbstractState{T} end

const Covariance{T} = Array{T,2} where T


"""
Discrete-time state.
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
Uncertain discrete-time state.
"""
mutable struct UncertainDiscreteState{T<:AbstractFloat} <:
    AbstractUncertainState{T}

    x::Vector{T}
    P::Covariance{T}
    t::Int64

    function UncertainDiscreteState(x::Vector{T}, P::Covariance{T},
        t::Int64) where T
            if length(x) != size(P)[1] || length(x) != size(P)[2]
                error("Dimensions of covariance and state do not match.")
            end
            new{T}(x, P, t)
    end
    function UncertainDiscreteState(x::Vector{T}, P::Covariance{T}) where T
            if length(x) != size(P)[1] || length(x) != size(P)[2]
                error("Dimensions of covariance and state do not match.")
            end
            new{T}(x, P, 0)
    end
end
UncertainDiscreteState{T<:AbstractFloat}(x::T, P::T) =
    UncertainDiscreteState([x], reshape([P],1,1))
UncertainDiscreteState{T<:AbstractFloat}(x::T, P::T, t::Int64) =
    UncertainDiscreteState([x], reshape([P],1,1), t)

function DiscreteState{T}(state::UncertainDiscreteState{T})
    return DiscreteState(state.x, state.t)
end

function UncertainDiscreteState{T}(state::DiscreteState{T})
    return UncertainDiscreteState(state.x, zeros(length(state.x),
        length(state.x)), state.t)
end

function UncertainDiscreteState{T}(state::DiscreteState{T}, P::Covariance{T})
    return UncertainDiscreteState(state.x, P, state.t)
end


"""
Continuous-time state.
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
Uncertain continuous-time state.
"""
mutable struct UncertainContinuousState{T<:AbstractFloat} <:
    AbstractUncertainState{T}

    x::Vector{T}
    P::Covariance{T}
    t::T

    function UncertainContinuousState(x::Vector{T}, P::Covariance{T},
        t::T) where T
            if length(x) != size(P)[1] || length(x) != size(P)[2]
                error("Dimensions of covariance and state do not match.")
            end
            new{T}(x, P, t)
    end
    function UncertainContinuousState(x::Vector{T}, P::Covariance{T}) where T
            if length(x) != size(P)[1] || length(x) != size(P)[2]
                error("Dimensions of covariance and state do not match.")
            end
            new{T}(x, P, 0.0)
    end
end
UncertainContinuousState{T<:AbstractFloat}(x::T, P::T) =
    UncertainContinuousState([x], reshape([P],1,1))
UncertainContinuousState{T<:AbstractFloat}(x::T, P::T, t::T) =
    UncertainContinuousState([x], reshape([P],1,1), t)

function ContinuousState{T}(state::UncertainContinuousState{T})
    return ContinuousState(state.x, state.t)
end

function UncertainContinuousState{T}(state::ContinuousState{T})
    return UncertainContinuousState(state.x, zeros(length(state.x),
        length(state.x)), state.t)
end

function UncertainContinuousState{T}(state::ContinuousState{T},P::Covariance{T})
    return UncertainContinuousState(state.x, P, state.t)
end


"""
    make_uncertain(state::AbstractUncertainState[, P::Covariance{T}])

Create an uncertain copy of the input absolute state.
"""
make_uncertain(state::DiscreteState) = UncertainDiscreteState(state)
make_uncertain{T}(state::DiscreteState{T}, P::Covariance{T}) =
    UncertainDiscreteState(state, P)
make_uncertain(state::ContinuousState) = UncertainContinuousState(state)
make_uncertain{T}(state::ContinuousState{T}, P::Covariance{T}) =
    UncertainContinuousState(state, P)

"""
    make_absolute(state::AbstractUncertainState)

Create an absolute copy of the input uncertain state.
"""
make_absolute(state::UncertainDiscreteState) = DiscreteState(state)
make_absolute(state::UncertainContinuousState) = ContinuousState(state)


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
format.
"""
function sample{T}(state::UncertainDiscreteState{T})
    return DiscreteState(
        state.x+chol(Hermitian(state.P))*randn(T, length(state.x)), state.t)
end
function sample{T}(state::UncertainContinuousState{T})
    return ContinuousState(
        state.x+chol(Hermitian(state.P))*randn(T, length(state.x)), state.t)
end


"""
    distance(state1::AbstractState{T}, state2::AbstractState{T})
"""
function distance{T}(state1::AbstractState{T}, state2::AbstractState{T})
    return sqrt(sum((state1.x .- state2.x).^2))
end


"""
    mahalanobis(x::AbstractAbsoluteState{T}, y::AbstractUncertainState{T})
"""
function mahalanobis{T}(x::AbstractAbsoluteState{T},
                        y::AbstractUncertainState{T})
    diff_vec = x.x .- y.x
    return sqrt(diff_vec'*inv(y.P)*diff_vec)
end
"""
    mahalanobis(x::AbstractAbsoluteState{T}, y::AbstractAbsoluteState{T},
                P::Array{T,2})
"""
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
function Base.broadcast(::typeof(*), A::Matrix, state::AbstractState)
    return A * state
end


"""
State assignment operations.
"""
function Base.broadcast!(::typeof(identity), out::AbstractAbsoluteState,
                            state::AbstractAbsoluteState)
    out.x .= state.x
    out.t = state.t
    return nothing
end
function Base.broadcast!(::typeof(identity), out::AbstractUncertainState,
                            state::AbstractUncertainState)
    out.x .= state.x
    out.P .+ state.P
    out.t = state.t
    return nothing
end
