"""
Batche Estimator
"""

abstract type AbstractBatchEstimator <: Estimator end


"""
Batch least squares estimator
"""
immutable LeastSquaresEstimator{T,S<:AbstractUncertainState{T}} <: Estimator
    sys::LinearSystem{T}
    obs::LinearObserver{T}
    estimate::S
    measurements::Vector{S}

    function LeastSquaresEstimator(sys::LinearSystem{T}, obs::LinearObserver{T},
        estimate::S, measurements::Vector{S}) where {T,S}
        new{T,S}(sys, obs, estimate, measurements)
    end
    function LeastSquaresEstimator(sys::LinearSystem{T}, obs::LinearObserver{T},
                                   estimate::S) where {T,S}
        new{T,S}(sys, obs, estimate, Vector{S}[])
    end
end


"""
Add measurement to LeastSquaresEstimator for processing.
"""
function add{T,S}(bls::LeastSquaresEstimator{T,S}, measurement::S)
    push!(bls.measurements, measurement)
end
function add{T,S}(bls::LeastSquaresEstimator{T,S}, measurements::Vector{S})
    for idx = 1:length(measurements)
        push!(bls.measurements, measurements[idx])
    end
end
