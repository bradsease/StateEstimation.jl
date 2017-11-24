"""
Kalman Filter Types and Methods
"""

abstract type AbstractKalmanFilter{T,S} <: SequentialEstimator{T,S} end


"""
Standard Kalman Filter type
"""
immutable KalmanFilter{T,S<:AbstractUncertainState{T}} <: AbstractKalmanFilter{T,S}
    sys::LinearSystem{T}
    obs::LinearObserver{T}
    estimate::S

    function KalmanFilter(sys::LinearSystem{T}, obs::LinearObserver{T},
                          estimate::S) where {T, S<:AbstractUncertainState{T}}
        assert_compatibility(sys, estimate)
        assert_compatibility(obs, estimate)
        new{T,S}(sys, obs, estimate)
    end

    function KalmanFilter(sys::LinearSystem{T}, obs::LinearObserver{T},
                          estimate::S) where {T, S<:AbstractAbsoluteState{T}}
        assert_compatibility(sys, estimate)
        assert_compatibility(obs, estimate)
        KalmanFilter(sys, obs, make_uncertain(estimate))
    end
end


const DiscreteKalmanFilter{T} =
    KalmanFilter{T,UncertainDiscreteState{T}} where T
const ContinuousKalmanFilter{T} =
    KalmanFilter{T,UncertainContinuousState{T}} where T


"""
    process!(kf::KalmanFilter, z::AbstractAbsoluteState)

Kalman filter correction step.
"""
function process!{T}(kf::DiscreteKalmanFilter{T}, z::DiscreteState{T})
    predict!(kf.sys, kf.estimate, z.t)
    yk = predict(kf.obs, kf.estimate)
    Kk = kf.estimate.P*kf.obs.H'*inv(yk.P)
    kf.estimate.x += Kk*(z.x - yk.x)
    kf.estimate.P -= Kk*kf.estimate.P*Kk'
    return nothing
end
function process!{T}(kf::DiscreteKalmanFilter{T}, z::DiscreteState{T},
                     archive::EstimatorHistory{T})
    if length(archive.states) == 0
        push!(archive.states, deepcopy(kf.estimate))
    end
    predict!(kf.sys, kf.estimate, z.t)
    yk = predict(kf.obs, kf.estimate)
    residual = (z.x - yk.x)
    Kk = kf.estimate.P*kf.obs.H'*inv(yk.P)
    kf.estimate.x += Kk*residual
    kf.estimate.P -= Kk*kf.estimate.P*Kk'
    push!(archive.residuals, UncertainDiscreteState(residual, yk.P, z.t))
    return nothing
end
function process!{T}(kf::ContinuousKalmanFilter{T}, z::ContinuousState{T})
    predict!(kf.sys, kf.estimate, z.t)
    yk = predict(kf.obs, kf.estimate)
    Kk = kf.estimate.P*kf.obs.H'*inv(yk.P)
    kf.estimate.x += Kk*(z.x - yk.x)
    kf.estimate.P -= Kk*kf.estimate.P*Kk'
    return nothing
end
function process!{T}(kf::ContinuousKalmanFilter{T}, z::ContinuousState{T},
                     archive::EstimatorHistory{T})
    if length(archive.states) == 0
        push!(archive.states, deepcopy(kf.estimate))
    end
    predict!(kf.sys, kf.estimate, z.t)
    yk = predict(kf.obs, kf.estimate)
    residual = (z.x - yk.x)
    Kk = kf.estimate.P*kf.obs.H'*inv(yk.P)
    kf.estimate.x += Kk*residual
    kf.estimate.P -= Kk*kf.estimate.P*Kk'
    push!(archive.residuals, UncertainContinuousState(residual, yk.P, z.t))
    return nothing
end


"""
    simulate(kf::KalmanFilter, t)

Simulate next measurement for a Kalman filter.
"""
function simulate{T}(kf::KalmanFilter{T}, t)
    return sample(predict(kf.obs, predict(kf.sys, kf.estimate, t)))
end
