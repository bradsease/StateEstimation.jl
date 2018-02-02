#
# Extended Kalman Filter Types and Methods
#


"""
    ExtendedKalmanFilter(sys::NonlinearSystem, obs::NonlinearObserver,
                         estimate::AbstractState[, consider_states::Vector{UInt}])

Extended Kalman Filter type. The internal filter model takes on a specific form
depending on the type of the initial estimate. For a `DiscreteState`,

\$x_k = F(k, x_{k-1}) + w_k\$
\$y_k = H(k, x_k) + v_k\$

where \$w_k \\sim N(0, Q)\$ and \$v_k \\sim N(0, R)\$. For a `ContinuousState`,

\$\\dot{x}(t_k) = F(t_k, x(t_k)) + w(t_k)\$
\$y(t_k) = H(t_k, x(t_k)) + v(t_k)\$

Construction of an ExtendedKalmanFilter requires an initial estimate.
Internally, the initial estimate is an uncertain state type. The constructor
automatically converts absolute initial estimates to uncertain states with zero
covariance.

The `consider_states` input contains a list of indices of state elements to be
considered in the filtering process and not updated.
"""
immutable ExtendedKalmanFilter{T,S<:AbstractUncertainState{T}} <:
                                                       AbstractKalmanFilter{T,S}
    sys::NonlinearSystem{T}
    obs::NonlinearObserver{T}
    estimate::S
    consider_states::Vector{UInt16}

    function ExtendedKalmanFilter(sys::NonlinearSystem{T},
                                  obs::NonlinearObserver{T}, estimate::S,
                                  consider_states::Vector
                                  ) where {T, S<:AbstractUncertainState{T}}
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
ExtendedKalmanFilter(sys, obs, estimate) =
    ExtendedKalmanFilter(sys, obs, estimate, [])
ExtendedKalmanFilter(sys, obs, estimate::AbstractAbsoluteState, consider_states) =
    ExtendedKalmanFilter(sys, obs, make_uncertain(estimate), consider_states)


const DiscreteEKF{T} =
    ExtendedKalmanFilter{T,UncertainDiscreteState{T}} where T
const ContinuousEKF{T} =
    ExtendedKalmanFilter{T,UncertainContinuousState{T}} where T


"""
    kalman_predict(kf::KalmanFilter, t)
"""
function kalman_predict(ekf::ExtendedKalmanFilter, t)
    xk = predict(ekf.estimate, ekf.sys, t)
    yk = predict(xk, ekf.obs)
    Pxy = xk.P * ekf.obs.dH_dx(t, ekf.estimate.x)'
    return xk, yk, Pxy
end
