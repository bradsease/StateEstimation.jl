using StateEstimation
using Base.Test

# Sanity check
@test 1 == 1


# Create linear observer
linear_sys = LinearSystem(0.5*eye(2), 0.001*eye(2))
linear_obs = LinearObserver(eye(2), 0.001*eye(2))
initial_est = UncertainDiscreteState([1.0, 2.0], 0.1*eye(2))
kf = KalmanFilter(linear_sys, linear_obs, initial_est)

for i = 1:10
    println(kf.estimate)
    measurement = simulate(kf)
    process!(kf, measurement)
end
println(kf.estimate)


println("***")
println("***")

# Create linear observer
linear_sys = LinearSystem(-eye(2), 0.001*eye(2))
linear_obs = LinearObserver(eye(2), 0.001*eye(2))
initial_est = UncertainContinuousState([1.0, 2.0], 0.1*eye(2))
kf = KalmanFilter(linear_sys, linear_obs, initial_est)
arc = EstimatorHistory()

for i = 1:10
    println(kf.estimate)
    measurement = simulate(kf, kf.estimate.t+0.1)
    process!(kf, measurement, arc)
end
println(kf.estimate)
