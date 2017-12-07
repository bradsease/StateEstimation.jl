#
# General systems
#

using DifferentialEquations

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

    ``x_k = F(k, x_{k-1}) + w`` where ``w ~ N(0, Q)``.

Continuous Form:

    ``x'(t) = F(t, x(t)) + w`` where ``w ~ N(0, Q)``.

Constructors:

    NonlinearSystem(F::Function, dF_dx::Function, Q::Covariance, ndim)

    NonlinearSystem(F::Function, dF_dx::Function, Q::AbstractFloat, ndim)
"""
struct NonlinearSystem{T<:AbstractFloat} <: AbstractSystem{T}
    F::Function
    dF_dx::Function
    Q::Covariance{T}

    function NonlinearSystem(F::Function, dF_dx::Function, Q::Covariance{T},
                             ndim::Integer) where T
        if (ndim, ndim) != size(Q)
            throw(DimensionMismatch("Incompatible size of system matrices."))
        end
        new{T}(F, dF_dx, Q)
    end
end
NonlinearSystem(F::Function, dF_dx::Function, Q::AbstractFloat, ndim::Integer) =
    NonlinearSystem(F, dF_dx, reshape([Q], 1, 1), ndim)


"""
    assert_compatibility(sys::AbstractSystem{T}, state::AbstractState{T})

Require system dimensions to be compatible with input state.
"""
function assert_compatibility{T}(sys::LinearSystem{T},state::AbstractState{T})
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
    solve_linear_system(A, x0, dt)
    solve_linear_system(A, k, x0, dt)
"""
function solve_linear_system(A::AbstractMatrix, x0::AbstractVector,
                             dt::AbstractFloat)
    return expm(A*dt)*x0
end
function solve_linear_system(A::AbstractMatrix, k::AbstractVector,
                             x0::AbstractVector, dt::AbstractFloat)
    n = length(x0)
    A_expanded = zeros(2*n, 2*n)
    A_expanded[1:n, 1:n] .= A
    A_expanded[1:n, n+1:end] .= eye(n)
    return (expm(A_expanded*dt)*[x0; k])[1:n]
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

    return reshape(solve_linear_system(Ap, Q[:], P[:], dt), size(P))
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

function predict!{T}(sys::NonlinearSystem{T}, state::DiscreteState{T},
                     t::Integer)
    for t_step = state.t:t-1
        state.x = sys.F(t_step, state.x)
    end
    state.t = t
    return nothing
end
function predict!{T}(sys::NonlinearSystem{T}, state::UncertainDiscreteState{T},
                     t::Integer)
    for t_step = state.t:t-1
        state.x = sys.F(t_step, state.x)
        jac = sys.dF_dx(t_step, state.x)
        state.P = jac*state.P*jac' + sys.Q
    end
    state.t = t
    return nothing
end
function predict!{T}(sys::NonlinearSystem{T}, state::ContinuousState{T}, t)
    solution = DifferentialEquations.solve(
        ODEProblem(sys.F, state.x, (state.t, t)), save_everystep=false)
    state.x .= solution.u[end]
    state.t = t
    return nothing
end
function predict!{T}(sys::NonlinearSystem{T},
                     state::UncertainContinuousState{T}, t)
    n = length(state.x)
    function combined_ode(t, x_in)
        x = x_in[1:n]
        P = reshape(x_in[n+1:end], n, n)
        xdot = sys.F(t, x)
        A = sys.dF_dx(t, x)
        Pdot = A*P + P*A' + sys.Q
        return [xdot; Pdot[:]]
    end

    combined_state = [state.x; state.P[:]]
    solution = DifferentialEquations.solve(ODEProblem(
        combined_ode, combined_state, (state.t, t)), save_everystep=false,
        reltol=1e-8, abstol=1e-8)

    state.x .= solution.u[end][1:n]
    state.P .= reshape(solution.u[end][n+1:end], n, n)
    state.t = t
    return nothing
end



#"""
#    simulate(sys::LinearSystem{T}, state::AbstractUncertainState{T})
#"""
#function simulate{T}(sys::LinearSystem{T}, state::AbstractUncertainState{T})
#    return sample(predict(sys, state))
#end
