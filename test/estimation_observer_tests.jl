using StateEstimation
using Base.Test

# Sanity check
@test 1 == 1


# Create linear observer
linear_obs = LinearObserver(eye(2), eye(2))
@test linear_obs.H == eye(2)
@test linear_obs.R == eye(2)

# Predict discrete state through linear observer
discrete_state = DiscreteState(ones(2))
result = predict(linear_obs, discrete_state)

# Test observability methods
linear_sys = LinearSystem(Float64[[1, -2] [-3, -4]]', 0.1*eye(2))
linear_obs = LinearObserver(reshape(Float64[1, 2], 1, 2), reshape([0.1], 1, 1))
@test observable(linear_sys, linear_obs) == false
@test observable(eye(2), eye(2))
