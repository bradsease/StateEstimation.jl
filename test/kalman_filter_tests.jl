using StateEstimation
using Base.Test

# Sanity check
@test 1 == 1


# Create linear observer
linear_sys = LinearSystem(0.5*eye(2), 0.001*eye(2))
linear_obs = LinearObserver(eye(2), 0.001*eye(2))
initial_est = UncertainDiscreteState([1.0, 2.0], 0.1*eye(2))

kf = KalmanFilter(linear_sys, linear_obs, initial_est)

for i = 1:20
    println(kf.estimate)
    measurement = simulate(kf)
    process!(kf, measurement)
end
println(kf.estimate)
