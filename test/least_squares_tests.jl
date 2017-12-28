# Least Squares Tests


# Test absolute state constructors
linear_sys = LinearSystem(1.0, 1.0)
linear_obs = LinearObserver(1.0, 1.0)
initial_est = ContinuousState(1.0)
lse = LeastSquaresEstimator(linear_sys, linear_obs, initial_est)
@test typeof(lse.estimate) <: UncertainContinuousState
initial_est = DiscreteState(1.0)
lse = LeastSquaresEstimator(linear_sys, linear_obs, initial_est)
@test typeof(lse.estimate) <: UncertainDiscreteState


# Test continuous-time linear least squares
srand(1)
linear_sys = LinearSystem(-eye(2), 0.001*eye(2))
linear_obs = LinearObserver(eye(2), 0.01*eye(2))
initial_est = UncertainContinuousState([1.0, 2.0], 0.1*eye(2))
true_state = initial_est
lse = LeastSquaresEstimator(linear_sys, linear_obs, initial_est)
for idx = 1:10
    true_state, measurement = simulate(lse.sys, lse.obs, true_state,(idx-1)*0.1)
    add!(lse, measurement)
end
@test distance(solve(lse), initial_est) < 0.2


# Test continuous-time linear least squares with archiving
srand(1)
linear_sys = LinearSystem(-eye(2), 0.001*eye(2))
linear_obs = LinearObserver(eye(2), 0.01*eye(2))
initial_est = UncertainContinuousState([1.0, 2.0], 0.1*eye(2))
true_state = initial_est
lse = LeastSquaresEstimator(linear_sys, linear_obs, initial_est)
for idx = 1:10
    true_state, measurement = simulate(lse.sys, lse.obs, true_state,(idx-1)*0.1)
    add!(lse, measurement)
end
archive = EstimatorHistory()
@test distance(solve(lse, archive), initial_est) < 0.2
@test (length(archive.states) == 10) & (length(archive.residuals) == 10)


# Test discrete-time linear least squares
srand(1)
linear_sys = LinearSystem(eye(3), 0.001*eye(3))
linear_obs = LinearObserver(eye(3), 0.01*eye(3))
initial_est = UncertainDiscreteState([1.0, 2.0, 3.0], 0.1*eye(3))
true_state = initial_est
lse = LeastSquaresEstimator(linear_sys, linear_obs, initial_est)
measurements = []
for idx = 1:10
    true_state, measurement = simulate(lse.sys, lse.obs, true_state, idx-1)
    push!(measurements, measurement)
end
add!(lse, measurements)
solve!(lse)
@test distance(lse.estimate, initial_est) < 0.2


# Test discrete-time linear least squares with archiving
srand(1)
linear_sys = LinearSystem(eye(3), 0.001*eye(3))
linear_obs = LinearObserver(eye(3), 0.01*eye(3))
initial_est = UncertainDiscreteState([1.0, 2.0, 3.0], 0.1*eye(3))
true_state = initial_est
lse = LeastSquaresEstimator(linear_sys, linear_obs, initial_est)
measurements = []
for idx = 1:10
    true_state, measurement = simulate(lse.sys, lse.obs, true_state, idx-1)
    push!(measurements, measurement)
end
add!(lse, measurements)
archive = EstimatorHistory()
solve!(lse, archive)
@test distance(lse.estimate, initial_est) < 0.2
@test (length(archive.states) == 10) & (length(archive.residuals) == 10)
