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
        if (size(H,1), size(H,1)) != size(R)
            throw(DimensionMismatch("Incompatible size of system matrices."))
        end
        new{T}(H, R)
    end

    function LinearObserver(H::Matrix{T}) where T
        new{T}(H, zeros(size(H)))
    end
end
LinearObserver{T<:AbstractFloat}(H::T) =
    LinearObserver(reshape([H],1,1), zeros(1,1))
LinearObserver{T<:AbstractFloat}(H::T, R::T) =
    LinearObserver(reshape([H],1,1), reshape([R],1,1))
LinearObserver{T<:AbstractFloat}(H::Vector{T}, R::Covariance{T}) =
    LinearObserver(reshape(H,length(H),1), R)
LinearObserver{T<:AbstractFloat}(H::RowVector{T}, R::T) =
    LinearObserver(Array(H), reshape([R],1,1))
LinearObserver{T<:AbstractFloat}(H::RowVector{T}, R::Covariance{T}) =
    LinearObserver(Array(H), R)


"""
Nonlinear observer.

    ``y_k = H(k, x_{k-1}) + v`` where ``v ~ N(0, R)``.

Constructors:

    NonlinearObserver(H::Function, dH_dx::Function, R::Covariance, ndim)

    NonlinearObserver(H::Function, dH_dx::Function, R::AbstractFloat, ndim)
"""
struct NonlinearObserver{T<:AbstractFloat} <: AbstractObserver{T}
    H::Function
    dH_dx::Function
    R::Covariance{T}

    function NonlinearObserver(H::Function, dH_dx::Function,
                               R::Covariance{T}) where T
        new{T}(H, dH_dx, R)
    end
end
NonlinearObserver(H::Function, dH_dx::Function, R::AbstractFloat) =
    NonlinearObserver(H, dH_dx, reshape([R], 1, 1))


"""
    assert_compatibility(obs::LinearObserver{T}, state::AbstractState{T})

Require linear observer dimensions to be compatible with input state.
"""
function assert_compatibility{T}(obs::LinearObserver{T},state::AbstractState{T})
    if size(obs.H, 2) != length(state.x)
        throw(DimensionMismatch("Observer incompatible with input state."))
    end
end
"""
    assert_compatibility(state::AbstractState{T}, obs::LinearObserver{T})

Require linear observer dimensions to be left-compatible with input state.
"""
function assert_compatibility{T}(state::AbstractState{T},obs::LinearObserver{T})
    if size(obs.H, 1) != length(state.x)
        throw(DimensionMismatch("Observer incompatible with input state."))
    end
end


"""
    observable(A, H)

Evaluate the observability of a linear model.
"""
function observable(A, H)
    n = size(A, 2)
    p = size(H, 1)
    observability = zeros(n*p, n)

    observability[1:p, :] .= H
    for idx = 1:n-1
        observability[p*idx+1:p*(idx+1), :] .= H * A
    end

    if rank(observability) == n
        return true
    else
        return false
    end
end
"""
    observable(sys::LinearSystem{T}, obs::LinearObserver{T})
"""
function observable{T}(sys::LinearSystem{T}, obs::LinearObserver{T})
    return observable(sys.A, obs.H)
end


"""
    predict(sys::LinearObserver{T}, state::AbstractState{T})
"""
function predict{T}(sys::LinearObserver{T}, state::DiscreteState{T})
    return DiscreteState(sys.H*state.x, state.t)
end
function predict{T}(sys::LinearObserver{T}, state::ContinuousState{T})
    return ContinuousState(sys.H*state.x, state.t)
end
function predict{T}(sys::LinearObserver{T}, state::UncertainDiscreteState{T})
    return UncertainDiscreteState(sys.H*state.x, sys.H*state.P*sys.H'+sys.R,
                                  state.t)
end
function predict{T}(sys::LinearObserver{T}, state::UncertainContinuousState{T})
    return UncertainContinuousState(sys.H*state.x, sys.H*state.P*sys.H'+sys.R,
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
