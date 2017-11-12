"""
General observers
"""

abstract type AbstractObserver{T} end


"""
Linear observer.

``y_k = H x_k + v``

where ``v ~ N(0, R)``.
"""
struct LinearObserver{T<:AbstractFloat} <: AbstractObserver{T}
    H::Matrix{T}
    R::Covariance{T}

    function LinearObserver(H::Matrix{T}, R::Covariance{T}) where T
        if size(H) != size(R)
            error("Incompatible size of system matrices.")
        end
        new{T}(H, R)
    end

    function LinearObserver(H::Matrix{T}) where T
        new{T}(H, zeros(size(H)))
    end
end

"""
    predict(sys::LinearObserver{T}, state::AbstractState{T})
"""
function predict{T}(sys::LinearObserver{T}, state::DiscreteState{T})
    return DiscreteState(sys.H*state.x, state.t)
end
function predict{T}(sys::LinearObserver{T}, state::UncertainDiscreteState{T})
    return UncertainDiscreteState(sys.H*state.x, sys.H*state.P*sys.H'+sys.R,
                                  state.t)
end

"""
    measure(sys::LinearObserver{T}, state::AbstractState{T})

Construct a measurement of an input state from a linear observer.
"""
function measure{T}(sys::LinearObserver{T}, state::DiscreteState{T})
    return DiscreteState(sys.H*state.x, state.t)
end
function measure{T}(sys::LinearObserver{T}, state::UncertainDiscreteState{T})
    return DiscreteState(sys.H*state.x, state.t)
end
function measure{T}(sys::LinearObserver{T}, state::ContinuousState{T})
    return ContinuousState(sys.H*state.x, state.t)
end
function measure{T}(sys::LinearObserver{T}, state::UncertainContinuousState{T})
    return ContinuousState(sys.H*state.x, state.t)
end
