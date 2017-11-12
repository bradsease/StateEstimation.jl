"""
General systems
"""

abstract type AbstractSystem{T} end


"""
Linear system of equations.

``x_k = A x_{k-1} + w``

where ``w ~ N(0, Q)``.
"""
struct LinearSystem{T<:AbstractFloat} <: AbstractSystem{T}
    A::Matrix{T}
    Q::Covariance{T}

    function LinearSystem(A::Matrix{T}, Q::Covariance{T}) where T
        if size(A) != size(Q)
            error("Incompatible size of system matrices.")
        end
        if size(A,1) != size(A,2)
            error("System 'A' matrix must be square.")
        end
        new{T}(A, Q)
    end

    function LinearSystem(A::Matrix{T}) where T
        new{T}(A, zeros(size(A)))
    end
end

"""
    assert_compatibility(sys::LinearSystem{T}, state::AbstractState{T})

Require linear system dimensions to be compatible with input state.
"""
function assert_compatibility{T}(sys::LinearSystem{T}, state::AbstractState{T})
    if size(sys.A, 2) != length(state.x)
       error("Linear system incompatible with input state.")
    end
end

"""
    state_transition_matrix(sys::LinearSystem{T}, state::AbstractState{T}, t)

Get state transition matrix for a LinearSystem over a specified time
span. Supports both discrete and continuous states in absolute and uncertain
forms. For discrete systems, t must be an integer.
"""
function state_transition_matrix{T}(sys::LinearSystem{T},
                                    state::DiscreteState{T}, t::Integer)
    return sys.A^(t-state.t)
end
function state_transition_matrix{T}(sys::LinearSystem{T},
                                    state::UncertainDiscreteState{T},t::Integer)
    return sys.A^(t-state.t)
end
function state_transition_matrix{T}(sys::LinearSystem{T},
                                    state::ContinuousState{T}, t::T)
    return expm(sys.A*(t-state.t))
end
function state_transition_matrix{T}(sys::LinearSystem{T},
                                    state::UncertainContinuousState{T}, t::T)
    return expm(sys.A*(t-state.t))
end


"""
    predict(sys::LinearSystem{T}, state::AbstractState{T})

Predict a state through a linear system.
"""
function predict{T}(sys::LinearSystem{T}, state::DiscreteState{T})
    return DiscreteState(sys.A*state.x, state.t+1)
end
function predict{T}(sys::LinearSystem{T}, state::UncertainDiscreteState{T})
    return UncertainDiscreteState(sys.A*state.x, sys.A*state.P*sys.A'+sys.Q,
                                  state.t+1)
end
function predict{T}(sys::LinearSystem{T}, state::ContinuousState{T}, t::T)
    return ContinuousState(expm(sys.A*(t-state.t))*state.x, t)
end
function predict{T}(sys::LinearSystem{T}, state::UncertainContinuousState{T},
                    t::T)
    A = expm(sys.A*(t-state.t))
    return UncertainContinuousState(A*state.x, A*state.P*A'+sys.Q, t)
end


"""
    predict!(sys::LinearSystem{T}, state::AbstractState{T})

In-place prediction of a state through a linear system.
"""
function predict!{T}(sys::LinearSystem{T}, state::DiscreteState{T})
    state.x = sys.A*state.x
    state.t += 1
    return nothing
end
function predict!{T}(sys::LinearSystem{T}, state::UncertainDiscreteState{T})
    state.x = sys.A*state.x
    state.P = sys.A*state.P*sys.A'+sys.Q
    state.t += 1
    return nothing
end
function predict!{T}(sys::LinearSystem{T}, state::ContinuousState{T}, t::T)
    state.x = expm(sys.A*(t-state.t))*state.x
    state.t = t
    return nothing
end
function predict!{T}(sys::LinearSystem{T}, state::UncertainContinuousState{T},
                    t::T)
    A = expm(sys.A*(t-state.t))
    state.x = A*state.x
    state.P = A*state.P*A'+sys.Q
    state.t = t
    return nothing
end



"""
    predict(sys::AbstractSystem{T}, state::DiscreteState{T}, t::Integer)

General multi-step discrete-time prediction.
"""
function predict{T}(sys::AbstractSystem{T}, state::DiscreteState{T}, t::Integer)
    if t < state.t
        error("Discrete states can only be predicted forward in time.")
    end

    out_state = deepcopy(state)
    for idx = 1:t-state.t
        predict!(sys, out_state)
    end

    return out_state
end
function predict{T}(sys::AbstractSystem{T}, state::UncertainDiscreteState{T},
                    t::Integer)
    if t < state.t
        error("Discrete states can only be predicted forward in time.")
    end

    out_state = deepcopy(state)
    for idx = 1:t-state.t
        predict!(sys, out_state)
    end

    return out_state
end


"""
    predict!(sys::AbstractSystem{T}, state::DiscreteState{T}, t::Integer)

Multi-step in-place discrete-time prediction.
"""
function predict!{T}(sys::AbstractSystem{T}, state::UncertainDiscreteState{T},
                     t::Integer)
    if t < state.t
        error("Discrete states can only be predicted forward in time.")
    end

    for idx = 1:t-state.t
        predict!(sys, state)
    end

    return nothing
end
function predict!{T}(sys::AbstractSystem{T}, state::DiscreteState{T},
                     t::Integer)
    if t < state.t
        error("Discrete states can only be predicted forward in time.")
    end

    for idx = 1:t-state.t
        predict!(sys, state)
    end

    return nothing
end


#"""
#    simulate(sys::LinearSystem{T}, state::AbstractUncertainState{T})
#"""
#function simulate{T}(sys::LinearSystem{T}, state::AbstractUncertainState{T})
#    return sample(predict(sys, state))
#end
