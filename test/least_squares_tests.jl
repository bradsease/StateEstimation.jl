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
linear_sys = LinearSystem(-eye(2), 0.0001*eye(2))
linear_obs = LinearObserver(eye(2), 0.01*eye(2))
initial_est = UncertainContinuousState([1.0, 2.0], 0.1*eye(2))
lse = LeastSquaresEstimator(linear_sys, linear_obs, initial_est)
simulator = make_simulator(lse)
correct_solution = deepcopy(simulator.true_state)
for idx = 1:10
    true_state, measurement = simulate(simulator, (idx-1)*0.1)
    add!(lse, measurement)
end
@test mahalanobis(correct_solution, solve(lse)) < 3


# Test continuous-time linear least squares with archiving
srand(1)
linear_sys = LinearSystem(-eye(2), 0.0001*eye(2))
linear_obs = LinearObserver(eye(2), 0.01*eye(2))
initial_est = UncertainContinuousState([1.0, 2.0], 0.1*eye(2))
lse = LeastSquaresEstimator(linear_sys, linear_obs, initial_est)
simulator = make_simulator(lse)
correct_solution = deepcopy(simulator.true_state)
for idx = 1:10
    true_state, measurement = simulate(simulator, (idx-1)*0.1)
    add!(lse, measurement)
end
archive = EstimatorHistory(lse)
@test mahalanobis(correct_solution, solve(lse, archive)) < 3
@test (length(archive.states) == 10) & (length(archive.residuals) == 10)


# Test discrete-time linear least squares
srand(1)
linear_sys = LinearSystem(eye(3), 0.0001*eye(3))
linear_obs = LinearObserver(eye(3), 0.01*eye(3))
initial_est = UncertainDiscreteState([1.0, 2.0, 3.0], 0.1*eye(3))
lse = LeastSquaresEstimator(linear_sys, linear_obs, initial_est)
simulator = make_simulator(lse)
correct_solution = deepcopy(simulator.true_state)
measurements = []
for idx = 1:10
    true_state, measurement = simulate(simulator, idx-1)
    push!(measurements, measurement)
end
add!(lse, measurements)
solve!(lse)
@test mahalanobis(correct_solution, lse.estimate) < 3


# Test discrete-time linear least squares with archiving
srand(1)
linear_sys = LinearSystem(eye(3), 0.0001*eye(3))
linear_obs = LinearObserver(eye(3), 0.01*eye(3))
initial_est = UncertainDiscreteState([1.0, 2.0, 3.0], 0.1*eye(3))
lse = LeastSquaresEstimator(linear_sys, linear_obs, initial_est)
simulator = make_simulator(lse)
correct_solution = deepcopy(simulator.true_state)
measurements = []
for idx = 1:10
    true_state, measurement = simulate(simulator, idx-1)
    push!(measurements, measurement)
end
add!(lse, measurements)
archive = EstimatorHistory(lse)
solve!(lse, archive)
@test mahalanobis(correct_solution, lse.estimate) < 3
@test (length(archive.states) == 10) & (length(archive.residuals) == 10)



# Test continuous-time nonlinear least squares
srand(1)
nl_sys_fcn(t, x::Vector) = -x;
nl_sys_jac(t, x::Vector) = -eye(length(x))
nonlin_sys = NonlinearSystem(nl_sys_fcn, nl_sys_jac, 0.0001*eye(2))
nl_obs_fcn(t, x::Vector) = x;
nl_obs_jac(t, x::Vector) = eye(length(x))
nonlin_obs = NonlinearObserver(nl_obs_fcn, nl_obs_jac, 0.01*eye(2))
initial_est = UncertainContinuousState([1.0, 3.0], eye(2))
nlse = NonlinearLeastSquaresEstimator(nonlin_sys, nonlin_obs, initial_est, 1e-4)
simulator = make_simulator(nlse)
correct_solution = deepcopy(simulator.true_state)
for idx = 1:10
    true_state, measurement = simulate(simulator, idx*0.1)
    add!(nlse, measurement)
end
solve!(nlse)
@test mahalanobis(correct_solution, nlse.estimate) < 3
