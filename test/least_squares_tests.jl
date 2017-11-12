using StateEstimation
using Base.Test

# Sanity check
@test 1 == 1


# Create linear observer
linear_sys = LinearSystem(-eye(2), 0.001*eye(2))
linear_obs = LinearObserver(eye(2), 0.001*eye(2))
initial_est = UncertainContinuousState([1.0, 2.0], 0.1*eye(2))

lse = LeastSquaresEstimator(linear_sys, linear_obs, initial_est)

for idx = 1:10
    add!(lse, simulate(lse, (idx-1)*0.1))
end