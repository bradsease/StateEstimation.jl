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
    add!(lse::LeastSquaresEstimator, measurement::AbstractAbsoluteState)
    add!(lse::LeastSquaresEstimator, measurements::Vector{AbstractAbsoluteState})

Add one or more measurements to a `LeastSquaresEstimator` for future processing.
The `LeastSquaresEstimator` will store these measurements internally.
"""
function add!(::LeastSquaresEstimator) end
function add!(lse::LeastSquaresEstimator, measurement::AbstractAbsoluteState)
    assert_compatibility(measurement, lse.obs)
    push!(lse.measurements, measurement)
    return nothing
end
function add!(lse::LeastSquaresEstimator, measurements::Vector)
    for idx = 1:length(measurements)
        add!(lse, measurements[idx])
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
                                 lse::LeastSquaresEstimator{T,S},
                                 initial_state::S)
    for idx = 1:length(lse.measurements)
        pred_measurement = predict(lse.obs,
            predict(lse.sys, initial_state, lse.measurements[idx].t))
        pred_measurement .-= lse.measurements[idx].x
        push!(residuals, pred_measurement)
    end
    return nothing
end
compute_residuals!(residuals::Vector, lse::LeastSquaresEstimator) =
    compute_residuals!(residuals, lse, lse.estimate)


"""
    compute_states(state_history::Vector, lse::LeastSquaresEstimator)

Compute states for all internal measurements, returning a Vector.
"""
function compute_states!{T,S}(state_history::Vector{S},
                              lse::LeastSquaresEstimator{T,S},
                              initial_state::S)
    for idx = 1:length(lse.measurements)
        pred_state = predict(lse.sys, initial_state, lse.measurements[idx].t)
        push!(state_history, pred_state)
    end
    return nothing
end
compute_states!(state_history::Vector, lse::LeastSquaresEstimator) =
    compute_states(state_history, lse, lse.estimate)


"""
    solve(lse:LeastSquaresEstimator[, archive::EstimatorHistory])

Solve for an updated state estimate using the internal model and measurements
in a `LeastSquaresEstimator`. Returns an uncertain state containing the updated
state estimate and its covariance. Use `solve!` to update the internal estimate
of the `LeastSquaresEstimator` in-place.

Optionally provide an `EstimatorHistory` archive variable to store intermediate
solution data.
"""
function solve{T}(lse::LeastSquaresEstimator{T})
    m = size(lse.obs.H, 1)
    n = length(lse.estimate.x)
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
function solve(lse::LeastSquaresEstimator, archive::EstimatorHistory)
    estimate = solve(lse)
    compute_residuals!(archive.residuals, lse, estimate)
    compute_states!(archive.states, lse, estimate)
    return estimate
end


"""
    solve!(lse:LeastSquaresEstimator[, archive::EstimatorHistory])
"""
function solve!(lse::LeastSquaresEstimator)
    lse.estimate .= solve(lse)
    return nothing
end
function solve!(lse::LeastSquaresEstimator, archive::EstimatorHistory)
    lse.estimate .= solve(lse, archive)
    return nothing
end
