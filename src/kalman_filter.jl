#
# Kalman Filter Types and Methods
#

abstract type AbstractKalmanFilter{T,S} <: SequentialEstimator{T,S} end

const AbstractDiscreteKalmanFilter{T} =
    AbstractKalmanFilter{T,UncertainDiscreteState{T}} where T
const AbstractContinuousKalmanFilter{T} =
    AbstractKalmanFilter{T,UncertainContinuousState{T}} where T

"""
    KalmanFilter(sys::LinearSystem, obs::LinearObserver,
                 estimate::AbstractState[, consider_states::Vector{UInt}])

Linear Kalman Filter type. The internal filter model takes on a specific form
depending on the type of the initial estimate. For a `DiscreteState`,

\$x_k = A x_{k-1} + w_k\$
\$y_k = H x_k + v_k\$

where \$w_k \\sim N(0, Q)\$ and \$v_k \\sim N(0, R)\$. For a `ContinuousState`,

\$\\dot{x}(t_k) = A x(t_k) + w(t_k)\$
\$y(t_k) = H x(t_k) + v(t_k)\$

Construction of a KalmanFilter requires an initial estimate. Internally, the
initial estimate is an uncertain state type. The constructor automatically
converts absolute initial estimates to uncertain states with zero covariance.

The `consider_states` input contains a list of indices of state elements to be
considered in the filtering process and not updated.
"""
immutable KalmanFilter{T,S<:AbstractUncertainState{T}} <: AbstractKalmanFilter{T,S}
    sys::LinearSystem{T}
    obs::LinearObserver{T}
    estimate::S
    consider_states::Vector{UInt16}

    function KalmanFilter(sys::LinearSystem{T}, obs::LinearObserver{T},
                          estimate::S, consider_states::Vector
                          ) where {T, S<:AbstractUncertainState{T}}
        assert_compatibility(sys, estimate)
        assert_compatibility(obs, estimate)
        if !allunique(consider_states)
            throw(ArgumentError("Consider state indices must be unique"))
        end
        for idx = 1:length(consider_states)
            if consider_states[idx] > length(estimate.x)
                throw(DimensionMismatch(
                    "Consider indices extend beyond length of initial state."))
            end
        end
        new{T,S}(sys, obs, estimate, consider_states)
    end
end
function KalmanFilter(sys::LinearSystem, obs::LinearObserver,
                      estimate::AbstractUncertainState)
    KalmanFilter(sys, obs, estimate, [])
end
function KalmanFilter(sys::LinearSystem, obs::LinearObserver,
                      estimate::AbstractAbsoluteState)
    KalmanFilter(sys, obs, make_uncertain(estimate))
end
function KalmanFilter(sys::LinearSystem, obs::LinearObserver,
                      estimate::AbstractAbsoluteState, consider_states::Vector)
    KalmanFilter(sys, obs, make_uncertain(estimate), consider_states)
end


const DiscreteKalmanFilter{T} =
    KalmanFilter{T,UncertainDiscreteState{T}} where T
const ContinuousKalmanFilter{T} =
    KalmanFilter{T,UncertainContinuousState{T}} where T

"""
    kalman_predict(kf::KalmanFilter, t)
"""
function kalman_predict(kf::KalmanFilter, t)
    xk = predict(kf.sys, kf.estimate, t)
    yk = predict(kf.obs, xk)
    Pxy = xk.P*kf.obs.H'
    return xk, yk, Pxy
end

"""
    kalman_update!(kf, yk, zk)
"""
function kalman_update!(kf::AbstractKalmanFilter, yk::AbstractUncertainState,
                        zk::AbstractState, Pxy::Matrix)
    Kk = Pxy*inv(yk.P)
    if isempty(kf.consider_states)
        kf.estimate.x += Kk*(zk.x - yk.x)
        kf.estimate.P -= Kk*yk.P*Kk'
    else
        prev_estimate = deepcopy(kf.estimate)
        kf.estimate.x += Kk*(zk.x - yk.x)
        kf.estimate.P -= Kk*yk.P*Kk'
        reset_consider_states!(prev_estimate, kf.estimate, kf.consider_states)
    end
end

"""
    reset_consider_states!(prev_state, new_state, consider_states)
"""
function reset_consider_states!(prev_state::AbstractUncertainState,
                                new_state::AbstractUncertainState,
                                consider_states::Vector)
   new_state.x[consider_states] .= prev_state.x[consider_states]
   for i = 1:length(new_state.x), j = 1:length(new_state.x)
       if (i in consider_states) && (j in consider_states)
           new_state.P[i,j] .= prev_state.P[i,j]
       end
   end
end


"""
    process!(kf::AbstractKalmanFilter, zk::AbstractAbsoluteState[, archive::EstimatorHistory])

Process a measurement with an arbitrary Kalman Filter. Predicts the filter's
internal estimate to the time of the input measurement and performs a
correction step.

The user may optionally provide an `EstimatorHistory` archive variable to store
the incremental state data produced during the process step.
"""
function process!(::AbstractKalmanFilter) end
function process!(kf::AbstractDiscreteKalmanFilter, zk::DiscreteState)
    xk, yk, Pxy = kalman_predict(kf, zk.t)
    kf.estimate .= xk
    kalman_update!(kf, yk, zk, Pxy)
    return nothing
end
function process!(kf::AbstractDiscreteKalmanFilter, zk::DiscreteState,
                  archive::EstimatorHistory)
    xk, yk, Pxy = kalman_predict(kf, zk.t)
    kf.estimate .= xk
    kalman_update!(kf, yk, zk, Pxy)

    if length(archive.states) == 0
        push!(archive.states, deepcopy(kf.estimate))
    end
    push!(archive.states, deepcopy(kf.estimate))
    push!(archive.residuals, UncertainDiscreteState(zk.x - yk.x, yk.P, zk.t))
    return nothing
end
function process!(kf::AbstractContinuousKalmanFilter, zk::ContinuousState)
    xk, yk, Pxy = kalman_predict(kf, zk.t)
    kf.estimate .= xk
    kalman_update!(kf, yk, zk, Pxy)
    return nothing
end
function process!(kf::AbstractContinuousKalmanFilter, zk::ContinuousState,
                  archive::EstimatorHistory)
    xk, yk, Pxy = kalman_predict(kf, zk.t)
    kf.estimate .= xk
    kalman_update!(kf, yk, zk, Pxy)

    if length(archive.states) == 0
        push!(archive.states, deepcopy(kf.estimate))
    end
    push!(archive.states, deepcopy(kf.estimate))
    push!(archive.residuals, UncertainContinuousState(zk.x - yk.x, yk.P, zk.t))
    return nothing
end
