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
                                   estimate::S) where {T,S}
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
function add!{T,S,M}(lse::LeastSquaresEstimator{T,S,M}, measurements::Vector{M})
    for idx = 1:length(measurements)
        add!(lse, measurements[idx])
    end
    return nothing
end


#"""
#    solve!(lse:LeastSquaresEstimator)
#"""
#function solve!(lse::LeastSquaresEstimator)
#
#end
#function solve!(lse::LeastSquaresEstimator{T}, archive::EstimatorHistory{T})
#
#end


#"""
#    solve(lse:LeastSquaresEstimator)
#"""
#function solve(lse::LeastSquaresEstimator)
#
# return solution, archive
#end



"""
    simulate(lse::LeastSquaresEstimator, t)

Simulate a measurement for a pre-configured LeastSquaresEstimator.
"""
function simulate{T}(lse::LeastSquaresEstimator{T}, t)
    return sample(predict(lse.obs, predict(lse.sys, lse.estimate, t)))
end
