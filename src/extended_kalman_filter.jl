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

#
# """
#     kalman_update!(kf, H, yk, zk)
# """
# function kalman_update!(kf::AbstractKalmanFilter, H::Matrix,
#                         yk::AbstractUncertainState, zk::AbstractState)
#     Kk = kf.estimate.P*H'*inv(yk.P)
#     if isempty(kf.consider_states)
#         kf.estimate.x += Kk*(zk.x - yk.x)
#         kf.estimate.P -= Kk*yk.P*Kk'
#     else
#         prev_estimate = deepcopy(kf.estimate)
#         kf.estimate.x += Kk*(zk.x - yk.x)
#         kf.estimate.P -= Kk*yk.P*Kk'
#         reset_consider_states!(prev_estimate, kf.estimate, kf.consider_states)
#     end
# end


"""
    process!(ekf::ExtendedKalmanFilter, zk::AbstractAbsoluteState)

Extended Kalman filter correction step.
"""
function process!{T}(ekf::DiscreteEKF{T}, zk::DiscreteState{T})
    predict!(ekf.sys, ekf.estimate, zk.t)
    kalman_update!(ekf, ekf.obs.dH_dx(zk.t, ekf.estimate.x),
                   predict(ekf.obs, ekf.estimate), zk)
    return nothing
end
# function process!{T}(kf::DiscreteKalmanFilter{T}, zk::DiscreteState{T},
#                      archive::EstimatorHistory{T})
#     predict!(kf.sys, kf.estimate, zk.t)
#     yk = predict(kf.obs, kf.estimate)
#     kalman_update!(kf, kf.obs.H, yk, zk)
#
#     if length(archive.states) == 0
#         push!(archive.states, deepcopy(kf.estimate))
#     end
#     push!(archive.states, deepcopy(kf.estimate))
#     push!(archive.residuals, UncertainDiscreteState(zk.x - yk.x, yk.P, zk.t))
#     return nothing
# end
# function process!{T}(kf::ContinuousKalmanFilter{T}, zk::ContinuousState{T})
#     predict!(kf.sys, kf.estimate, zk.t)
#     kalman_update!(kf, kf.obs.H, predict(kf.obs, kf.estimate), zk)
#     return nothing
# end
# function process!{T}(kf::ContinuousKalmanFilter{T}, zk::ContinuousState{T},
#                      archive::EstimatorHistory{T})
#     predict!(kf.sys, kf.estimate, zk.t)
#     yk = predict(kf.obs, kf.estimate)
#     kalman_update!(kf, kf.obs.H, yk, zk)
#
#     if length(archive.states) == 0
#         push!(archive.states, deepcopy(kf.estimate))
#     end
#     push!(archive.states, deepcopy(kf.estimate))
#     push!(archive.residuals, UncertainContinuousState(zk.x - yk.x, yk.P, zk.t))
#     return nothing
# end


"""
    simulate(kf::ExtendedKalmanFilter, t)

Simulate next measurement for an Extended Kalman filter.

TODO: Implement more accurate approach. This method is only a linear
approximation.
"""
function simulate(ekf::ExtendedKalmanFilter, t)
    return sample(predict(ekf.obs, predict(ekf.sys, ekf.estimate, t)))
end
