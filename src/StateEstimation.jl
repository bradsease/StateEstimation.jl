"""
"""
module StateEstimation


export Estimator, Filter, AbstractSystem, AbstractObserver, DiscreteState,
       UncertainDiscreteState, ContinuousState, UncertainContinuousState,
       LinearSystem, predict, predict!, LinearObserver, measure, simulate,
       process!, make_absolute, make_uncertain, sample, KalmanFilter,
       EstimatorHistory, plot_archive, LeastSquaresEstimator, add!

abstract type Estimator end
abstract type Filter <: Estimator end


include("states.jl")
include("systems.jl")
include("observers.jl")
include("archive.jl")
include("least_squares.jl")
include("kalman_filter.jl")
#include("extended_kalman_filter.jl")
#include("unscented_kalman_filter.jl")

end
