"""
Kalman Filter Types and Methods
"""

abstract type AbstractKalmanFilter <: Filter end


"""
Standard Kalman Filter type
"""
immutable KalmanFilter{T,S<:AbstractUncertainState{T}} <: AbstractKalmanFilter
    sys::LinearSystem{T}
    obs::LinearObserver{T}
    estimate::S

    function KalmanFilter(sys::LinearSystem{T}, obs::LinearObserver{T},
                          estimate::S) where {T, S<:AbstractUncertainState{T}}

         if size(sys.A, 2) != length(estimate.x)
             error("Linear system incompatible with input state.")
         end
         if size(obs.H, 2) != length(estimate.x)
             error("Linear system incompatible with input state.")
         end
         new{T,S}(sys, obs, estimate)
    end
end

const DiscreteKalmanFilter{T} =
    KalmanFilter{T,UncertainDiscreteState{T}} where T
const ContinuousKalmanFilter{T} =
    KalmanFilter{T,UncertainContinuousState{T}} where T


"""
    predict!(kf::KalmanFilter)

Kalman filter prediction step.
"""
function predict!{T}(kf::DiscreteKalmanFilter{T})
    predict!(kf.sys, kf.estimate)
    return nothing
end
function predict!{T}(kf::KalmanFilter{T}, t)
    predict!(kf.sys, kf.estimate, t)
    return nothing
end


"""
    correct!(kf::KalmanFilter, z::AbstractState)

Kalman filter correction step.
"""
function correct!{T}(kf::DiscreteKalmanFilter{T}, z::DiscreteState{T})
    if kf.estimate.t != z.t
        error("Measurement does not correspond to current discrete-time step.")
    end
    yk = predict(kf.obs, kf.estimate)
    Kk = kf.estimate.P*kf.obs.H'*inv(yk.P)
    kf.estimate.x .+= Kk*(z.x - yk.x)
    kf.estimate.P .-= Kk*kf.estimate.P*Kk'
    return nothing
end
function correct!{T}(kf::ContinuousKalmanFilter{T}, z::ContinuousState{T})
    if kf.estimate.t != z.t
        error("Measurement does not correspond to current time step.")
    end
    yk = predict(kf.obs, kf.estimate)
    Kk = kf.estimate.P*kf.obs.H'*inv(yk.P)
    kf.estimate.x .+= Kk*(z.x - yk.x)
    kf.estimate.P .-= Kk*kf.estimate.P*Kk'
    return nothing
end



"""
    process!(kf::KalmanFilter)
"""
function process!{T}(kf::KalmanFilter{T}, z::AbstractState{T})
    predict!(kf, z.t)
    correct!(kf, z)
    return nothing
end


"""
    simulate(kf::KalmanFilter, t)

Simulate next measurement for a Kalman filter.
"""
function simulate{T}(kf::KalmanFilter{T}, t)
    return sample(predict(kf.obs, predict(kf.sys, kf.estimate, t)))
end
"""
    simulate(kf::DiscreteKalmanFilter)

Simulate next measurement for a discrete Kalman filter.
"""
function simulate{T}(kf::DiscreteKalmanFilter{T})
    return sample(predict(kf.obs, predict(kf.sys, kf.estimate)))
end
