"""
General observers
"""

abstract type AbstractObserver{T} end

"""
    LinearObserver(H::Matrix[, R::Matrix])
    LinearObserver(H::AbstractFloat[, R::AbstractFloat])
    LinearObserver(H::Vector[, R::Matrix])
    LinearObserver(H::RowVector[, R::Union{Matrix, AbstractFloat}])

Linear observer with the form

\$y_k = H x_k + v_k \\quad \\text{where} \\quad  v_k \\sim N(0, R)\$

A linear observer can be constructed with both matrix and scalar inputs. The
observer will not model measurement noise if the covariance R is not provided.
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
end
LinearObserver{T}(H::Matrix{T}) = LinearObserver(H,zeros(T,size(H,1),size(H,1)))
LinearObserver{T<:AbstractFloat}(H::T) =
    LinearObserver(reshape([H],1,1), zeros(T,1,1))
LinearObserver{T<:AbstractFloat}(H::T, R::T) =
    LinearObserver(reshape([H],1,1), reshape([R],1,1))

LinearObserver(H::Vector) = LinearObserver(reshape(H,length(H),1))
LinearObserver{T<:AbstractFloat}(H::Vector{T}, R::Covariance{T}) =
    LinearObserver(reshape(H,length(H),1), R)

LinearObserver(H::RowVector) = LinearObserver(Array(H))
LinearObserver{T<:AbstractFloat}(H::RowVector{T}, R::T) =
    LinearObserver(Array(H), reshape([R],1,1))
LinearObserver{T<:AbstractFloat}(H::RowVector{T}, R::Covariance{T}) =
    LinearObserver(Array(H), R)


"""
    NonlinearObserver(H::Function, dH_dx::Function, R::Matrix)
    NonlinearObserver(H::Function, dH_dx::Function, R::AbstractFloat)

Nonlinear observer with the form

\$y_k = H(k, x_k) + v \\quad  \\text{where} \\quad v \\sim N(0, R)\$

NonlinearObserver constructors require both the measurement function,
`H(t::Number, x::Vector)`, and its Jacobian, `dH_dx(t::Number, x::Vector)`. Both
`H()` and `dH_dx()` must return a vector.
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
    observable(A::Matrix, H::Matrix)
    observable(sys::LinearSystem, obs::LinearObserver)

Evaluate the observability of a linear model. Returns a boolean True/False
result.
"""
function observable(A, H)
    n = size(A, 2)
    p = size(H, 1)
    observability = zeros(n*p, n)

    observability[1:p, :] .= H
    for idx = 1:n-1
        observability[p*idx+1:p*(idx+1), :] .= H * A
    end

    rank(observability) == n ? true : false
end
observable(sys::LinearSystem, obs::LinearObserver) = observable(sys.A, obs.H)


"""
    predict(obs::AbstractObserver, state::AbstractState)

Predict a state through an arbitrary observer. The `predict` method will advance
both the state vector and, if necessary, its covariance through the input
measurement model. For NonlinearObservers, the covariance prediction is only a
linear approximation.
"""
function predict(::AbstractObserver) end
function predict{T}(obs::LinearObserver{T}, state::DiscreteState{T})
    return DiscreteState(obs.H*state.x, state.t)
end
function predict{T}(obs::LinearObserver{T}, state::ContinuousState{T})
    return ContinuousState(obs.H*state.x, state.t)
end
function predict{T}(obs::LinearObserver{T}, state::UncertainDiscreteState{T})
    return UncertainDiscreteState(
        obs.H*state.x, obs.H*state.P*obs.H'+obs.R, state.t)
end
function predict{T}(obs::LinearObserver{T}, state::UncertainContinuousState{T})
    return UncertainContinuousState(
        obs.H*state.x, obs.H*state.P*obs.H'+obs.R, state.t)
end
function predict{T}(obs::NonlinearObserver{T}, state::DiscreteState{T})
    return DiscreteState(obs.H(state.t, state.x), state.t)
end
function predict{T}(obs::NonlinearObserver{T}, state::ContinuousState{T})
    return ContinuousState(obs.H(state.t, state.x), state.t)
end
function predict{T}(obs::NonlinearObserver{T}, state::UncertainDiscreteState{T})
    jac = obs.dH_dx(state.t, state.x)
    return UncertainDiscreteState(
        obs.H(state.t, state.x), jac*state.P*jac'+obs.R, state.t)
end
function predict{T}(obs::NonlinearObserver{T},
                    state::UncertainContinuousState{T})
    jac = obs.dH_dx(state.t, state.x)
    return UncertainContinuousState(
        obs.H(state.t, state.x), jac*state.P*jac'+obs.R, state.t)
end


"""
   simulate(obs::AbstractObserver, state::AbstractState)

Simulate a state observation with initial state error and process noise. This
method returns an absolute state by sampling the initial state (if uncertain),
and propagating that state through the input system in the presence of
measurement noise.
"""
function simulate(::AbstractObserver) end
function simulate(obs::AbstractObserver, state::AbstractAbsoluteState)
    return simulate(obs, make_uncertain(state))
end
function simulate(obs::LinearObserver, state::AbstractUncertainState)
    return sample(predict(obs, state))
end
function simulate(obs::NonlinearObserver,state::AbstractUncertainState)
    return sample(predict(obs, make_uncertain(sample(state))))
end
