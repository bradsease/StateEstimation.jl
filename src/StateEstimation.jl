"""
"""
module StateEstimation


export Estimator, Filter, AbstractSystem, AbstractObserver, DiscreteState,
       UncertainDiscreteState, ContinuousState, UncertainContinuousState,
       LinearSystem, predict, predict!, LinearObserver, measure, simulate,
       process!, make_absolute, make_uncertain, sample, KalmanFilter

abstract type Estimator end
abstract type Filter <: Estimator end


include("states.jl")
include("systems.jl")
include("observers.jl")
#include("batch_estimator.jl")
include("kalman_filter.jl")
#include("unscented_kalman_filter.jl")

end
