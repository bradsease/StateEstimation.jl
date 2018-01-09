using Base.Test
using StateEstimation


println("1. Testing State constructors and methods...")
include("state_tests.jl")

println("2. Testing System constructors and methods...")
include("system_tests.jl")

println("3. Testing Observer constructors and methods...")
include("observer_tests.jl")

println("4. Testing UnscentedTransform constructors and methods...")
include("unscented_transform_tests.jl")

println("5. Testing KalmanFilter constructors and methods...")
include("kalman_filter_tests.jl")

println("6. Testing ExtendedKalmanFilter constructors and methods...")
include("extended_kalman_filter_tests.jl")

println("7. Testing MultiTargetFilter constructors and methods...")
include("multi_target_filter_tests.jl")

println("8. Testing LeastSquaresEstimator constructors and methods...")
include("least_squares_tests.jl")
