using Base.Test
using StateEstimation


println("Testing State constructors and methods...")
include("state_tests.jl")

println("Testing System constructors and methods...")
include("system_tests.jl")

println("Testing Observer constructors and methods...")
include("observer_tests.jl")

println("Testing KalmanFilter constructors and methods...")
include("kalman_filter_tests.jl")

println("Testing ExtendedKalmanFilter constructors and methods...")
include("extended_kalman_filter_tests.jl")

println("Testing MultiTargetFilter constructors and methods...")
include("multi_target_filter_tests.jl")

println("Testing LeastSquaresEstimator constructors and methods...")
include("least_squares_tests.jl")
