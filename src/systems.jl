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
function LinearSystem{T<:AbstractFloat}(A::T, Q::T)
    return LinearSystem(reshape([A], 1, 1), reshape([Q], 1, 1))
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
function predict{T}(sys::LinearSystem{T}, state::AbstractAbsoluteState{T}, t)
    out_state = state_transition_matrix(sys, state, t) * state
    out_state.t = t
    return out_state
end
function predict{T}(sys::LinearSystem{T}, state::AbstractUncertainState{T}, t)
    out_state = state_transition_matrix(sys, state, t) * state
    out_state.P .+= sys.Q
    out_state.t = t
    return out_state
end


"""
    predict!(sys::LinearSystem{T}, state::AbstractState{T})

In-place prediction of a state through a linear system.
"""
function predict!{T}(sys::LinearSystem{T}, state::AbstractAbsoluteState{T}, t)
    state .= state_transition_matrix(sys, state, t) * state
    state.t = t
    return nothing
end
function predict!{T}(sys::LinearSystem{T}, state::AbstractUncertainState{T}, t)
    state .= state_transition_matrix(sys, state, t) * state
    state.P .+= sys.Q
    state.t = t
    return nothing
end


#"""
#    simulate(sys::LinearSystem{T}, state::AbstractUncertainState{T})
#"""
#function simulate{T}(sys::LinearSystem{T}, state::AbstractUncertainState{T})
#    return sample(predict(sys, state))
#end
