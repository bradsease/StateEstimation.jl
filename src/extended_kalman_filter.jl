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
    kalman_update!(kf, H, yk, zk)
"""
function kalman_update!(ekf::ExtendedKalmanFilter, yk::AbstractUncertainState,
                        zk::AbstractState)
    Kk = ekf.estimate.P*ekf.obs.dH_dx(zk.t, ekf.estimate.x)'*inv(yk.P)
    if isempty(ekf.consider_states)
        ekf.estimate.x += Kk*(zk.x - yk.x)
        ekf.estimate.P -= Kk*yk.P*Kk'
    else
        prev_estimate = deepcopy(ekf.estimate)
        ekf.estimate.x += Kk*(zk.x - yk.x)
        ekf.estimate.P -= Kk*yk.P*Kk'
        reset_consider_states!(prev_estimate, ekf.estimate, ekf.consider_states)
    end
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
