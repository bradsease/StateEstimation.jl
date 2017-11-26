using Base.Test
using StateEstimation


include("estimation_state_tests.jl")
include("estimation_system_tests.jl")
include("estimation_observer_tests.jl")

include("kalman_filter_tests.jl")
include("multi_target_filter_tests.jl")

include("least_squares_tests.jl")
