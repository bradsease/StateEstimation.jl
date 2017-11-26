"""
General systems
"""

abstract type AbstractSystem{T} end


"""
Linear system of equations.

Discrete Form:

    ``x_k = A x_{k-1} + w`` where ``w ~ N(0, Q)``.

Continuous Form:

    ``x'(t) = A x(t) + w`` where ``w ~ N(0, Q)``.

Constructors:

    LinearSystem(A::Matrix, Q::Covariance)

    LinearSystem(A::Matrix)

    LinearSystem(A::AbstractFloat)

    LinearSystem(A::AbstractFloat, Q::AbstractFloat)
"""
struct LinearSystem{T<:AbstractFloat} <: AbstractSystem{T}
    A::Matrix{T}
    Q::Covariance{T}

    function LinearSystem(A::Matrix{T}, Q::Covariance{T}) where T
        if size(A) != size(Q)
            throw(DimensionMismatch("Incompatible size of system matrices."))
        end
        if size(A,1) != size(A,2)
            throw(DimensionMismatch("System 'A' matrix must be square."))
        end
        new{T}(A, Q)
    end

    function LinearSystem(A::Matrix{T}) where T
        new{T}(A, zeros(T, size(A)))
    end
end
LinearSystem{T<:AbstractFloat}(A::T) =
    LinearSystem(reshape([A], 1, 1), zeros(T, 1, 1))
LinearSystem{T<:AbstractFloat}(A::T, Q::T) =
    LinearSystem(reshape([A], 1, 1), reshape([Q], 1, 1))


"""
Nonlinear system of equations

Discrete Form:

    ``x_k = F(x_{k-1}, k) + w`` where ``w ~ N(0, Q)``.

Continuous Form:

    ``x'(t) = F(x(t), t) + w`` where ``w ~ N(0, Q)``.

Constructors:

    NonLinearSystem(F::Function, dF_dx::Function, Q::Covariance, ndim)

    NonLinearSystem(F::Function, dF_dx::Function, Q::AbstractFloat, ndim)
"""
struct NonLinearSystem{T<:AbstractFloat} <: AbstractSystem{T}
    F::Function
    dF_dx::Function
    Q::Covariance{T}

    function NonLinearSystem(F::Function, dF_dx::Function, Q::Covariance{T},
                             ndim::Integer) where T
        if (ndim, ndim) != size(Q)
            throw(DimensionMismatch("Incompatible size of system matrices."))
        end
        new{T}(F, dF_dx, Q)
    end
end
NonLinearSystem(F::Function, dF_dx::Function, Q::AbstractFloat, ndim::Integer) =
    NonLinearSystem(F, dF_dx, reshape([Q], 1, 1), ndim)


"""
    assert_compatibility(sys::LinearSystem{T}, state::AbstractState{T})

Require linear system dimensions to be compatible with input state.
"""
function assert_compatibility{T}(sys::LinearSystem{T}, state::AbstractState{T})
    if size(sys.A, 2) != length(state.x)
        throw(DimensionMismatch("Linear system incompatible with input state."))
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
function predict{T}(sys::AbstractSystem{T}, state::AbstractState{T}, t)
    out_state = deepcopy(state)
    predict!(sys, out_state, t)
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
function predict!{T}(sys::LinearSystem{T}, state::UncertainDiscreteState{T},
                     t::Integer)
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

function predict!{T}(sys::NonLinearSystem{T}, state::DiscreteState{T},
                     t::Integer)
    for t_step = state.t:t-1
        sys.F(state.x, t_step)
    end
    state.t = t
    return nothing
end
function predict!{T}(sys::NonLinearSystem{T}, state::UncertainDiscreteState{T},
                     t::Integer)
    for t_step = state.t:t-1
        sys.F(state.x, t_step)
        jac = sys.dF_dx(state.x, t_step)
        state.P = jac*state.P*jac' + sys.Q
    end
    state.t = t
    return nothing
end



#"""
#    simulate(sys::LinearSystem{T}, state::AbstractUncertainState{T})
#"""
#function simulate{T}(sys::LinearSystem{T}, state::AbstractUncertainState{T})
#    return sample(predict(sys, state))
#end
