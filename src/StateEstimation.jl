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
    sample, distance, mahalanobis, expand_state, reduce_state,
    # Systems
    AbstractSystem, LinearSystem, NonlinearSystem, predict, predict!,
    state_transition_matrix,
    # Observers
    AbstractObserver, LinearObserver, NonlinearObserver, observable,
    # Data archiving
    EstimatorHistory, plot_state_history, plot_residuals,
    # Kalman filters
    AbstractKalmanFilter, KalmanFilter, ExtendedKalmanFilter, simulate,
    process!, kalman_filter, kalman_filter!, extended_kalman_filter,
    extended_kalman_filter!,
    # Unscented Kalman filter
    UnscentedTransform, compute_weights, compute_sigma_points, transform,
    trasnform!, augment, UnscentedKalmanFilter, unscented_kalman_filter,
    unscented_kalman_filter!,
    # Least squares estimation
    LeastSquaresEstimator, NonlinearLeastSquaresEstimator, add!, solve, solve!,
    # Multi-target filtering
    MultiTargetFilter, NearestNeighborMTF,
    # Simulation
    SingleStateSimulator, MultiStateSimulator, make_simulator


abstract type Estimator{T,S} end
abstract type SequentialEstimator{T,S} <: Estimator{T,S} end
abstract type BatchEstimator{T,S} <: Estimator{T,S} end


include("states.jl")
include("systems.jl")
include("observers.jl")
include("archive.jl")

include("kalman_filter.jl")
include("extended_kalman_filter.jl")
include("unscented_transform.jl")
#include("unscented_kalman_filter.jl")
include("multi_target_filter.jl")

include("least_squares.jl")

include("simulator.jl")

end
