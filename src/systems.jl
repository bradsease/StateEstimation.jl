#
# General systems
#

using DifferentialEquations

abstract type AbstractSystem{T} end


"""
    LinearSystem(A::Matrix[, Q::Matrix])
    LinearSystem(A::AbstractFloat[, Q::AbstractFloat])

Linear system of equations. Systems take a specific form depending on the type
of state they are used in conjunction with. With a `DiscreteState`, the system
takes on a discrete form

\$x_k = A x_{k-1} + w_{k-1} \\quad \\text{where} \\quad w_{k-1} \\sim N(0, Q)\$

With a `ContinuousState`, the system takes on a continous ODE form

\$\\dot{x}(t) = A x(t) + w(t)\$

A LinearSystem can be constructed with both matrix and scalar inputs. The
linear system will not contain process noise if the covaraince Q is
not provided.
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
end
LinearSystem{T<:AbstractFloat}(A::Matrix{T}) = LinearSystem(A, zeros(T,size(A)))
LinearSystem{T<:AbstractFloat}(A::T) =
    LinearSystem(reshape([A], 1, 1), zeros(T, 1, 1))
LinearSystem{T<:AbstractFloat}(A::T, Q::T) =
    LinearSystem(reshape([A], 1, 1), reshape([Q], 1, 1))


"""
    NonlinearSystem(F::Function, dF_dx::Function, Q::Covariance[, tolerances::Tuple])
    NonlinearSystem(F::Function, dF_dx::Function, Q::AbstractFloat[, tolerances::Tuple])

Nonlinear system of equations. Systems take a specific form depending on the
type of state they are used in conjunction with. With a `DiscreteState`, the
system takes on a discrete form

\$x_k = F(k-1, x_{k-1}) + w_{k-1} \\quad \\text{where} \\quad w_{k-1} \\sim N(0, Q)\$

With a `ContinuousState`, the system takes on a continous ODE form

\$\\dot{x}(t) = F(t, x(t)) + w(t) \\quad \\text{where} \\quad w(t) \\sim N(0, Q)\$

NonlinearSystem constructors require both the function,
`F(t::Number, x::Vector)`, and its Jacobian, `dF_dx(t::Number, x::Vector)`. Both
`F()` and `dF_dx()` must return a vector.

The user may optionally specify the integrator tolerances with a tuple of
`(abstol, reltol)`. The default tolerances are `(1e-2, 1e-2)`.
"""
struct NonlinearSystem{T<:AbstractFloat} <: AbstractSystem{T}
    F::Function
    dF_dx::Function
    Q::Covariance{T}
    abstol::T
    reltol::T

    function NonlinearSystem(F::Function, dF_dx::Function, Q::Covariance{T},
                             predict_tolerances=(1e-2, 1e-2)) where T
        new{T}(F, dF_dx, Q, predict_tolerances[1], predict_tolerances[2])
    end
end
NonlinearSystem(F::Function, dF_dx::Function, Q::AbstractFloat,
    predict_tolerances=(1e-2, 1e-2)) =
    NonlinearSystem(F, dF_dx, reshape([Q], 1, 1), predict_tolerances)


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
function state_transition_matrix{T,S<:UnionDiscrete{T}}(sys::NonlinearSystem{T},
                                                        state::S, t::Integer)
    x = state.x
    stm = eye(length(x))
    for t = state.t:t-1
        stm = sys.dF_dx(t, x)*stm
        x = sys.F(t, x)
    end
    #return DiscreteState(x, t), stm
    return stm
end
function state_transition_matrix{T,S<:UnionContinuous{T}}(
    sys::NonlinearSystem{T}, state::S, t::T)
    function stm_ode(t, u, du)
        du[:,1] .= sys.F(t, u[:,1])
        du[:,2:end] .= sys.dF_dx(t, u[:,1])*u[:,2:end]
    end
    initial_state = hcat(state.x, eye(T, length(state.x)))
    problem = ODEProblem(stm_ode, initial_state, (state.t, t))
    solution = DifferentialEquations.solve(problem, abstol=sys.abstol,
                                           reltol=sys.reltol)
    return solution.u[end][:, 2:end]
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
    predict(sys::AbstractSystem, state::AbstractState, t::Number)

Predict a state through an arbitrary system. The time step, `t`, must match the
input state in its type, i.e. for a DiscreteState, `t` must be an integer.

The `predict` method will advance both the state vector and, if necessary, its
covariance through the dynamics of the input system. For NonlinearSystems, the
covariance prediction is only a linear approximation.

Use `predict!` to update the input state with the predicted state in-place.
"""
function predict(::AbstractSystem) end
function predict{T}(sys::AbstractSystem{T}, state::AbstractState{T}, t)
    out_state = deepcopy(state)
    predict!(sys, out_state, t)
    return out_state
end


"""
    predict!(sys::AbstractSystem, state::AbstractState, t::Number)

In-place prediction of a state through an arbitrary system.
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
    if t != state.t
        solution = DifferentialEquations.solve(
            ODEProblem(sys.F, state.x, (state.t, t)),
            save_everystep=false, abstol=sys.abstol, reltol=sys.reltol)
        state.x .= solution.u[end]
        state.t = t
    end
    return nothing
end
function predict!{T}(sys::NonlinearSystem{T},
                     state::UncertainContinuousState{T}, t)
    if t != state.t
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
        solution = DifferentialEquations.solve(
            ODEProblem(combined_ode, combined_state, (state.t, t)),
            save_everystep=false, abstol=sys.abstol, reltol=sys.reltol)

        state.x .= solution.u[end][1:n]
        state.P .= reshape(solution.u[end][n+1:end], n, n)
        state.t = t
    end
    return nothing
end



"""
    simulate(sys::AbstractSystem, state::AbstractState, t::Real)

Simulate a state prediction with initial state error and process noise. This
method returns an absolute state by sampling the initial state (if uncertain),
and propagating that state through the input system in the presence of process
noise.
"""
function simulate(::AbstractSystem) end
function simulate(sys::AbstractSystem, state::AbstractAbsoluteState, t::Real)
    return simulate(sys, make_uncertain(state), t)
end
function simulate(sys::LinearSystem, state::AbstractUncertainState, t::Real)
    return sample(predict(sys, state, t))
end
function simulate(sys::NonlinearSystem, state::UncertainDiscreteState, t::Integer)
    sampled_initial_state = sample(state)
    return sample(predict(sys, make_uncertain(sampled_initial_state), t))
end
function simulate{T}(sys::NonlinearSystem{T},
                     state::UncertainContinuousState{T}, t::T)
    sampled_initial_state = sample(state)
    if all(sys.Q .== 0)
        simulated_state = predict(sys, sampled_initial_state, t)
    else
        f(t, u, du) = du .= sys.F(t, u)
        g(t, u, du) = du .= chol(Hermitian(sys.Q))*u
        prob = SDEProblem(f, g, sampled_initial_state.x, (state.t, t))
        soln = DifferentialEquations.solve(prob)
        simulated_state = ContinuousState(soln.u[end], t)
    end
    return simulated_state
end
