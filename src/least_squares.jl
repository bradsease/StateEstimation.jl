#
# Least Squares Estimation
#


"""
    LeastSquaresEstimator(sys::LinearSystem, obs::LinearObserver,
                          estimate::AbstractUncertainState)

Linear batch least squares estimator. The internal estimator model takes on a
specific form depending on the type of the initial estimate. For a
`DiscreteState`,

\$x_k = A x_{k-1} + w_k\$
\$y_k = H x_k + v_k\$

where \$w_k \\sim N(0, Q)\$ and \$v_k \\sim N(0, R)\$. For a `ContinuousState`,

\$\\dot{x}(t_k) = A x(t_k) + w(t_k)\$
\$y(t_k) = H x(t_k) + v(t_k)\$

Construction of a LeastSquaresEstimator requires an initial estimate.
Internally, the initial estimate is an uncertain state type. The constructor
automatically converts absolute initial estimates to uncertain states with zero
covariance.

`LeastSquaresEstimator` does not currently use the value of the initial estimate
or its covariance in solving for a new estimate.
"""
immutable LeastSquaresEstimator{T,S<:AbstractUncertainState{T},
                            M<:AbstractAbsoluteState{T}} <: BatchEstimator{T,S}
    sys::LinearSystem{T}
    obs::LinearObserver{T}
    estimate::S
    measurements::Vector{M}

    function LeastSquaresEstimator(sys::LinearSystem{T}, obs::LinearObserver{T},
        estimate::S) where {T,S<:AbstractUncertainState{T}}
        assert_compatibility(sys, estimate)
        assert_compatibility(obs, estimate)
        M = absolute_type(estimate)
        new{T,S,M}(sys, obs, estimate, Vector{M}([]))
    end
end
function LeastSquaresEstimator(sys::LinearSystem, obs::LinearObserver,
                               estimate::AbstractAbsoluteState)
    LeastSquaresEstimator(sys, obs, make_uncertain(estimate))
end


"""
    NonlinearLeastSquaresEstimator(sys::NonlinearSystem, obs::NonlinearObserver,
        estimate::AbstractState[, tolerance::AbstractFloat, max_iterations::Integer])
"""
immutable NonlinearLeastSquaresEstimator{T,S<:AbstractUncertainState{T},
                            M<:AbstractAbsoluteState{T}} <: BatchEstimator{T,S}
    sys::NonlinearSystem{T}
    obs::NonlinearObserver{T}
    estimate::S
    measurements::Vector{M}

    tolerance::T
    max_iterations::UInt16

    function NonlinearLeastSquaresEstimator(sys::NonlinearSystem{T},
        obs::NonlinearObserver{T}, estimate::S, tol=1e-2, max_iterations=15,
        ) where {T,S<:AbstractUncertainState{T}}
        M = absolute_type(estimate)
        new{T,S,M}(sys, obs, estimate, Vector{M}([]), tol, max_iterations)
    end
end
function NonlinearLeastSquaresEstimator(sys::NonlinearSystem,
    obs::NonlinearObserver, estimate::AbstractAbsoluteState,
    tol=1e-2, max_iterations=15)
    NonlinearLeastSquaresEstimator(sys, obs, make_uncertain(estimate), tol)
end


"""
    add!(estimator::BatchEstimator, measurement::AbstractAbsoluteState)
    add!(estimator::BatchEstimator, measurements::Vector{AbstractAbsoluteState})

Add one or more measurements to a `BatchEstimator` for future processing. The
`BatchEstimator` will store these measurements internally.
"""
function add!(::BatchEstimator) end
function add!(estimator::BatchEstimator, measurement::AbstractAbsoluteState)
    assert_compatibility(measurement, estimator.obs)
    push!(estimator.measurements, measurement)
    return nothing
end
function add!(estimator::BatchEstimator, measurements::Vector)
    for idx = 1:length(measurements)
        add!(estimator, measurements[idx])
    end
    return nothing
end


"""
    compute_residuals!(residuals::Vector, lse::LeastSquaresEstimator[,
                       initial_state::AbstractUncertainState])

Compute residuals for all internal measurements, returning a Vector of
residual states.
"""
function compute_residuals!{T,S}(residuals::Vector{S},
                                 estimator::BatchEstimator{T,S},
                                 initial_state::S)
    for idx = 1:length(estimator.measurements)
        pred_measurement = predict(estimator.obs, predict(
            estimator.sys, initial_state, estimator.measurements[idx].t))
        pred_measurement .-= estimator.measurements[idx].x
        push!(residuals, pred_measurement)
    end
    return nothing
end
compute_residuals!(residuals::Vector, estimator::BatchEstimator) =
    compute_residuals!(residuals, estimator, estimator.estimate)


"""
    compute_states(state_history::Vector, lse::LeastSquaresEstimator)

Compute states for all internal measurements, returning a Vector.
"""
function compute_states!{T,S}(state_history::Vector{S},
                              estimator::BatchEstimator{T,S},
                              initial_state::S)
    for idx = 1:length(estimator.measurements)
        pred_state = predict(
            estimator.sys, initial_state, estimator.measurements[idx].t)
        push!(state_history, pred_state)
    end
    return nothing
end
compute_states!(state_history::Vector, estimator::BatchEstimator) =
    compute_states(state_history, estimator, estimator.estimate)


"""
    solve(lse:LeastSquaresEstimator[, archive::EstimatorHistory])

Solve for an updated state estimate using the internal model and measurements
in a `LeastSquaresEstimator`. Returns an uncertain state containing the updated
state estimate and its covariance. Use `solve!` to update the internal estimate
of the `LeastSquaresEstimator` in-place.

Optionally provide an `EstimatorHistory` archive variable to store intermediate
solution data.
"""
function solve(::BatchEstimator) end
function solve{T}(lse::LeastSquaresEstimator{T})
    m,n = size(lse.obs.H)
    A = zeros(T, n, m*length(lse.measurements))
    b = zeros(T, m*length(lse.measurements))
    C = spzeros(T, m*length(lse.measurements), m*length(lse.measurements))

    for idx = 1:length(lse.measurements)
        start_idx = (idx-1)*m + 1
        end_idx = start_idx + m - 1

        A[:, start_idx:end_idx] .= state_transition_matrix(lse.sys,
            lse.estimate, lse.measurements[idx].t)' * lse.obs.H'
        b[start_idx:end_idx] .= lse.measurements[idx].x
        C[start_idx:end_idx, start_idx:end_idx] .= lse.obs.R
    end

    estimate = deepcopy(lse.estimate)
    temp = inv(A*A')*A
    estimate.x = temp * b
    estimate.P = temp * C * temp'

    return estimate
end
function solve{T}(nlse::NonlinearLeastSquaresEstimator{T})
    m,n = size(nlse.measurements[1].x, 1), length(nlse.estimate.x)
    A = zeros(T, n, m*length(nlse.measurements))
    b = zeros(T, m*length(nlse.measurements))
    C = spzeros(T, m*length(nlse.measurements), m*length(nlse.measurements))
    temp = zeros(T, n, m)

    estimate = deepcopy(nlse.estimate)
    convergence = nlse.tolerance+1
    iteration_count = nlse.max_iterations

    while convergence > nlse.tolerance && iteration_count > 0
        for idx = 1:length(nlse.measurements)
            start_idx = (idx-1)*m + 1
            end_idx = start_idx + m - 1

            t = nlse.measurements[idx].t
            predicted_measurement =
                predict(nlse.obs, predict(nlse.sys, estimate, t))

            A[:, start_idx:end_idx] .= state_transition_matrix(nlse.sys,
                estimate, t)' * nlse.obs.dH_dx(t, estimate.x)'
            b[start_idx:end_idx] .=
                nlse.measurements[idx].x - predicted_measurement.x
            C[start_idx:end_idx, start_idx:end_idx] .= nlse.obs.R
        end

        temp = inv(A*A')*A
        delta = temp * b
        estimate.x += delta
        convergence = norm(delta ./ estimate.x)
        iteration_count -= 1
    end

    estimate.P = temp * C * temp'
    return estimate
end
function solve(estimator::BatchEstimator, archive::EstimatorHistory)
    estimate = solve(estimator)
    compute_residuals!(archive.residuals, estimator, estimate)
    compute_states!(archive.states, estimator, estimate)
    return estimate
end


"""
    solve!(lse:LeastSquaresEstimator[, archive::EstimatorHistory])
"""
function solve!(estimator::BatchEstimator)
    estimator.estimate .= solve(estimator)
    return nothing
end
function solve!(estimator::BatchEstimator, archive::EstimatorHistory)
    estimator.estimate .= solve(estimator, archive)
    return nothing
end
