using Base.Test
using StateEstimation


include("state_tests.jl")
include("system_tests.jl")
include("observer_tests.jl")

include("kalman_filter_tests.jl")
include("multi_target_filter_tests.jl")

include("least_squares_tests.jl")
