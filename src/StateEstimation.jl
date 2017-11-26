"""
"""
module StateEstimation


export
    # Top-level abstract types
    Estimator, SequentialEstimator, BatchEstimator,
    # States
    AbstractState, AbstractAbsoluteState, AbstractUncertainState,
    DiscreteState, UncertainDiscreteState, ContinuousState,
    UncertainContinuousState, make_uncertain, make_absolute,
    sample, distance, mahalanobis,
    # Systems
    AbstractSystem, LinearSystem, NonLinearSystem, predict, predict!,
    state_transition_matrix,
    # Observers
    AbstractObserver, LinearObserver, measure, observable,
    # Data archiving
    EstimatorHistory, plot_archive,
    # Kalman filters
    AbstractKalmanFilter, KalmanFilter, simulate, process!,
    # Least squares estimation
    LeastSquaresEstimator, add!, solve, solve!,
    # Multi-target filtering
    MultiTargetFilter, NearestNeighborMTF


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
