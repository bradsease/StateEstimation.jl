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
