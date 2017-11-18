"""
"""
module StateEstimation


export Estimator, SequentialEstimator, AbstractSystem, AbstractObserver,
       DiscreteState, UncertainDiscreteState, ContinuousState,
       UncertainContinuousState, LinearSystem, predict, predict!,
       LinearObserver, measure, simulate, process!, make_absolute,
       make_uncertain, sample, KalmanFilter, EstimatorHistory, plot_archive,
       LeastSquaresEstimator, add!, NearestNeighborMTF, distance, mahalanobis,
       AbstractState, AbstractAbsoluteState, AbstractUncertainState,
       state_transition_matrix, observable, solve

abstract type Estimator end
abstract type SequentialEstimator{T,S} <: Estimator end
abstract type BatchEstimator{T} <: Estimator end


include("states.jl")
include("systems.jl")
include("observers.jl")
include("archive.jl")

include("kalman_filter.jl")
#include("extended_kalman_filter.jl")
#include("unscented_kalman_filter.jl")
include("multi_target_filter.jl")

include("least_squares.jl")

end
