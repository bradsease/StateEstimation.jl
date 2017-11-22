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
LinearSystem{T<:AbstractFloat}(A::T) =
    LinearSystem(reshape([A], 1, 1), zeros(1,1))
LinearSystem{T<:AbstractFloat}(A::T, Q::T) =
    LinearSystem(reshape([A], 1, 1), reshape([Q], 1, 1))


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
    assert_compatibility(sys, state)
    return sys.A^(t-state.t)
end
function state_transition_matrix{T}(sys::LinearSystem{T},
                                    state::UncertainDiscreteState{T},t::Integer)
    assert_compatibility(sys, state)
    return sys.A^(t-state.t)
end
function state_transition_matrix{T}(sys::LinearSystem{T},
                                    state::ContinuousState{T}, t::T)
    assert_compatibility(sys, state)
    return expm(sys.A*(t-state.t))
end
function state_transition_matrix{T}(sys::LinearSystem{T},
                                    state::UncertainContinuousState{T}, t::T)
    assert_compatibility(sys, state)
    return expm(sys.A*(t-state.t))
end


"""
    continuous_predict_cov(A::Matrix, Q::Matrix, P::Matrix, dt)

Predict covariance through a continuous linear system over a designated
time step.
"""
function continuous_predict_cov(A::Matrix, Q::Matrix, P::Matrix, dt)
    n = size(A,1)
    Ap = zeros(n*n, n*n)
    for i = 1:n
        shift = (i-1)*n
        @inbounds Ap[shift+1:shift+n, shift+1:shift+n] .= A
    end
    for i = 1:n*n, j = 1:n
        @inbounds Ap[(j-1)*n + (i-1)%n + 1, i] += A[j, 1+floor(Integer,(i-1)/n)]
    end
    if det(Ap) < eps()
        A_exp = expm(A*dt)
        return A_exp*P*A_exp' + dt*Q
    else
        Ap_exp = expm(Ap*dt)
        reshape(Ap_exp*P[:] + (Ap_exp - eye(size(Ap,1)))*inv(Ap)*Q[:], size(P))
    end
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
function predict{T}(sys::LinearSystem{T}, state::UncertainDiscreteState{T}, t)
    assert_compatibility(sys, state)
    out_state = deepcopy(state)
    for idx = state.t:t-1
        out_state.x .= sys.A*out_state.x
        out_state.P .= sys.A*out_state.P*sys.A' + sys.Q
    end
    out_state.t = t
    return out_state
end
function predict{T}(sys::LinearSystem{T}, state::UncertainContinuousState{T}, t)
    out_state = deepcopy(state)
    out_state.x = state_transition_matrix(sys, state, t) * state.x
    out_state.P = continuous_predict_cov(sys.A, sys.Q, state.P, t-state.t)
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
function predict!{T}(sys::LinearSystem{T}, state::UncertainDiscreteState{T}, t)
    assert_compatibility(sys, state)
    for idx = state.t:t-1
        state.x = sys.A*state.x
        state.P = sys.A*state.P*sys.A' + sys.Q
    end
    state.t = t
    return nothing
end
function predict!{T}(sys::LinearSystem{T}, state::UncertainContinuousState{T},t)
    state.x = state_transition_matrix(sys, state, t) * state.x
    state.P = continuous_predict_cov(sys.A, sys.Q, state.P, t-state.t)
    state.t = t
    return nothing
end



#"""
#    simulate(sys::LinearSystem{T}, state::AbstractUncertainState{T})
#"""
#function simulate{T}(sys::LinearSystem{T}, state::AbstractUncertainState{T})
#    return sample(predict(sys, state))
#end
