using StateEstimation
using Base.Test

# Test construction with absolute state
linear_sys = LinearSystem(0.5*eye(2), 0.001*eye(2))
linear_obs = LinearObserver(eye(2), 0.001*eye(2))
initial_est = DiscreteState([1.0, 2.0])
kf = KalmanFilter(linear_sys, linear_obs, initial_est)
@test typeof(kf.estimate) <: UncertainDiscreteState
@test kf.estimate.P == zeros(2,2)
@test_throws ArgumentError correct!(kf, DiscreteState(ones(2), 10))

# Test discrete kalman filter
srand(1);
linear_sys = LinearSystem(0.5*eye(2), 0.001*eye(2))
linear_obs = LinearObserver(eye(2), 0.001*eye(2))
initial_est = UncertainDiscreteState([1.0, 2.0], 0.1*eye(2))
kf = KalmanFilter(linear_sys, linear_obs, initial_est)
for i = 1:10
    measurement = simulate(kf)
    process!(kf, measurement)
end
@test_throws ArgumentError correct!(kf, DiscreteState(ones(2), 0))

# Test discrete kalman filter with archiving
srand(1);
linear_sys = LinearSystem(0.5*eye(2), 0.001*eye(2))
linear_obs = LinearObserver(eye(2), 0.001*eye(2))
initial_est = UncertainDiscreteState([1.0, 2.0], 0.1*eye(2))
kf = KalmanFilter(linear_sys, linear_obs, initial_est)
archive = EstimatorHistory()
for i = 1:10
    measurement = simulate(kf)
    process!(kf, measurement, archive)
end
@test_throws ArgumentError correct!(kf, DiscreteState(ones(2), 0))

# Test continuous kalman filter
srand(1);
linear_sys = LinearSystem(-eye(3), 0.001*eye(3))
linear_obs = LinearObserver(eye(3), 0.001*eye(3))
initial_est = UncertainContinuousState([1.0, 2.0, 3.0], 0.1*eye(3))
kf = KalmanFilter(linear_sys, linear_obs, initial_est)
for i = 1:10
    measurement = simulate(kf, kf.estimate.t+0.1)
    process!(kf, measurement)
end
@test_throws ArgumentError correct!(kf, ContinuousState(ones(3), 0.0))

# Test continuous kalman filter with archiving
srand(1);
linear_sys = LinearSystem(-eye(3), 0.001*eye(3))
linear_obs = LinearObserver(eye(3), 0.001*eye(3))
initial_est = UncertainContinuousState([1.0, 2.0, 3.0], 0.1*eye(3))
kf = KalmanFilter(linear_sys, linear_obs, initial_est)
archive = EstimatorHistory()
for i = 1:10
    measurement = simulate(kf, kf.estimate.t+0.1)
    process!(kf, measurement, archive)
end
@test_throws ArgumentError correct!(kf, ContinuousState(ones(3), 0.0))
