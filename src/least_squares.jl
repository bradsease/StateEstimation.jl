#
# Least Squares Estimation
#


"""
Batch least squares estimator
"""
immutable LeastSquaresEstimator{T,S<:AbstractUncertainState{T},
                            M<:AbstractAbsoluteState{T}} <: BatchEstimator{T}
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

    function LeastSquaresEstimator(sys::LinearSystem{T}, obs::LinearObserver{T},
        estimate::M) where {T,M<:AbstractAbsoluteState{T}}
        assert_compatibility(sys, estimate)
        assert_compatibility(obs, estimate)
        S = uncertain_type(estimate)
        new{T,S,M}(sys, obs, make_uncertain(estimate), Vector{M}([]))
    end
end


"""
    add!(lse:LeastSquaresEstimator, measurement::AbstractAbsoluteState)

Add measurement to LeastSquaresEstimator for processing.
"""
function add!{T,S,M}(lse::LeastSquaresEstimator{T,S,M}, measurement::M)
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
function compute_residuals!{T,S}(residuals::Vector,
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
function compute_states!{T,S}(state_history::Vector,
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
    solve(lse:LeastSquaresEstimator)
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
function solve{T}(lse::LeastSquaresEstimator{T}, archive::EstimatorHistory{T})
    estimate = solve(lse)
    compute_residuals!(archive.residuals, lse, estimate)
    compute_states!(archive.states, lse, estimate)
    return estimate
end


"""
    solve!(lse:LeastSquaresEstimator)
"""
function solve!{T}(lse::LeastSquaresEstimator{T})
    lse.estimate .= solve(lse)
    return nothing
end
function solve!{T}(lse::LeastSquaresEstimator{T}, archive::EstimatorHistory{T})
    lse.estimate .= solve(lse, archive)
    return nothing
end


"""
    simulate(lse::LeastSquaresEstimator, t)

Simulate a measurement for a pre-configured LeastSquaresEstimator.
"""
function simulate{T}(lse::LeastSquaresEstimator{T}, t)
    return sample(predict(lse.obs, predict(lse.sys, lse.estimate, t)))
end
