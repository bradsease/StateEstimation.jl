# Kalman Filter tests


# Test construction with absolute state
linear_sys = LinearSystem(0.5*eye(2), 0.001*eye(2))
linear_obs = LinearObserver(eye(2), 0.001*eye(2))
initial_est = DiscreteState([1.0, 2.0])
kf = KalmanFilter(linear_sys, linear_obs, initial_est)
@test typeof(kf.estimate) <: UncertainDiscreteState
@test kf.estimate.P == zeros(2,2)

# Test consider constructors
kf = KalmanFilter(linear_sys, linear_obs, initial_est, [2])
kf = KalmanFilter(linear_sys, linear_obs, make_uncertain(initial_est), [2])
@test_throws DimensionMismatch KalmanFilter(linear_sys, linear_obs, initial_est, [3])
@test_throws ArgumentError KalmanFilter(linear_sys, linear_obs, initial_est, [3,3])

# Test discrete kalman filter
srand(2);
linear_sys = LinearSystem(0.5*eye(2), 0.001*eye(2))
linear_obs = LinearObserver(eye(2), 0.001*eye(2))
initial_est = UncertainDiscreteState([1.0, 2.0], 0.1*eye(2))
kf = KalmanFilter(linear_sys, linear_obs, initial_est)
simulator = make_simulator(kf)
for i = 1:10
    true_state, measurement = simulate(simulator, i)
    process!(kf, measurement)
    @test mahalanobis(true_state, kf.estimate) < 3
end

# Test discrete consider kalman filter
srand(2);
A = [1.0 0.5 0.01; -0.5 1.0 0.01; 0.0 0.0 1.0]
linear_sys = LinearSystem(A, 0.001*eye(3))
linear_obs = LinearObserver(eye(2,3), 0.1*eye(2))
initial_est = UncertainDiscreteState([1.0, 2.0, 3.0], 0.1*eye(3))
consider_states = [3]
kf = KalmanFilter(linear_sys, linear_obs, initial_est, consider_states)
simulator = make_simulator(kf)
for i = 1:10
    true_state, measurement = simulate(simulator, i)
    process!(kf, measurement)
    @test mahalanobis(true_state, kf.estimate) < 3
end
@test kf.estimate.P[3,3] == initial_est.P[3,3]

# Test discrete kalman filter with archiving
srand(2);
linear_sys = LinearSystem(0.5*eye(2), 0.001*eye(2))
linear_obs = LinearObserver(eye(2), 0.01*eye(2))
initial_est = UncertainDiscreteState([1.0, 2.0], 0.1*eye(2))
kf = KalmanFilter(linear_sys, linear_obs, initial_est)
simulator = make_simulator(kf)
archive = EstimatorHistory(kf)
for i = 1:10
    true_state, measurement = simulate(simulator, i)
    process!(kf, measurement, archive)
    @test mahalanobis(true_state, kf.estimate) < 3
end

# Test continuous kalman filter
srand(2);
linear_sys = LinearSystem(-eye(3), 0.001*eye(3))
linear_obs = LinearObserver(eye(3), 0.01*eye(3))
initial_est = UncertainContinuousState([1.0, 2.0, 3.0], 0.1*eye(3))
kf = KalmanFilter(linear_sys, linear_obs, initial_est)
simulator = make_simulator(kf)
for i = 1:10
    true_state, measurement = simulate(simulator, kf.estimate.t+0.1)
    process!(kf, measurement)
    @test mahalanobis(true_state, kf.estimate) < 3
end

# Test continuous kalman filter with archiving
srand(2);
linear_sys = LinearSystem(-eye(3), 0.001*eye(3))
linear_obs = LinearObserver(eye(3), 0.01*eye(3))
initial_est = UncertainContinuousState([1.0, 2.0, 3.0], 0.1*eye(3))
kf = KalmanFilter(linear_sys, linear_obs, initial_est)
simulator = make_simulator(kf)
archive = EstimatorHistory(kf)
for i = 1:10
    true_state, measurement = simulate(simulator, kf.estimate.t+0.1)
    process!(kf, measurement, archive)
    @test mahalanobis(true_state, kf.estimate) < 3
end



# Test continuous kalman filter convenience function
srand(2);
linear_sys = LinearSystem(-eye(3), 0.001*eye(3))
linear_obs = LinearObserver(eye(3), 0.001*eye(3))
estimate = UncertainContinuousState([1.0, 2.0, 3.0], 0.1*eye(3))
simulator = make_simulator(KalmanFilter(linear_sys, linear_obs, estimate))
for i = 1:5
    true_state, measurement = simulate(simulator, kf.estimate.t+0.1)
    estimate = kalman_filter(estimate, linear_sys, linear_obs, measurement)
end

# Test continuous kalman filter with archiving
srand(2);
linear_sys = LinearSystem(-eye(3), 0.001*eye(3))
linear_obs = LinearObserver(eye(3), 0.001*eye(3))
estimate = UncertainContinuousState([1.0, 2.0, 3.0], 0.1*eye(3))
kf = KalmanFilter(linear_sys, linear_obs, estimate)
simulator = make_simulator(kf)
archive = EstimatorHistory(kf)
for i = 1:5
    true_state, measurement = simulate(simulator, kf.estimate.t+0.1)
    estimate =
        kalman_filter(estimate, linear_sys, linear_obs, measurement, archive)
end
