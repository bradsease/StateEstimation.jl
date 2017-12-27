#
# Extended Kalman Filter Types and Methods
#


"""
Extended Kalman Filter
"""
immutable ExtendedKalmanFilter{T,S<:AbstractUncertainState{T}} <:
                                                       AbstractKalmanFilter{T,S}
    sys::NonlinearSystem{T}
    obs::NonlinearObserver{T}
    estimate::S
    consider_states::Vector{UInt16}

    function ExtendedKalmanFilter(sys::NonlinearSystem{T},
                                  obs::NonlinearObserver{T}, estimate::S
                                  ) where {T, S<:AbstractUncertainState{T}}
        new{T,S}(sys, obs, estimate, [])
    end
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
function ExtendedKalmanFilter(sys::NonlinearSystem{T},
                              obs::NonlinearObserver{T}, estimate::S
                              ) where {T, S<:AbstractAbsoluteState{T}}
    ExtendedKalmanFilter(sys, obs, make_uncertain(estimate))
end
function ExtendedKalmanFilter(sys::NonlinearSystem{T},
                              obs::NonlinearObserver{T}, estimate::S,
                              consider_states::Vector
                              ) where {T, S<:AbstractAbsoluteState{T}}
    ExtendedKalmanFilter(sys, obs, make_uncertain(estimate), consider_states)
end


const DiscreteEKF{T} =
    ExtendedKalmanFilter{T,UncertainDiscreteState{T}} where T
const ContinuousEKF{T} =
    ExtendedKalmanFilter{T,UncertainContinuousState{T}} where T


"""
    kalman_predict(kf::KalmanFilter, t)
"""
function kalman_predict(ekf::ExtendedKalmanFilter, t)
    xk = predict(ekf.sys, ekf.estimate, t)
    yk = predict(ekf.obs, xk)
    Pxy = xk.P * ekf.obs.dH_dx(t, ekf.estimate.x)'
    return xk, yk, Pxy
end


"""
    simulate(kf::ExtendedKalmanFilter, t)

Simulate next measurement for an Extended Kalman filter.

TODO: Implement more accurate approach. This method is only a linear
approximation.
"""
function simulate(ekf::ExtendedKalmanFilter, t)
    return sample(predict(ekf.obs, predict(ekf.sys, ekf.estimate, t)))
end
