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
