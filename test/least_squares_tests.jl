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
lse = LeastSquaresEstimator(linear_sys, linear_obs, initial_est)
simulator = make_simulator(lse)
for idx = 1:10
    true_state, measurement = simulate(simulator, (idx-1)*0.1)
    add!(lse, measurement)
end
@test distance(solve(lse), initial_est) < 0.2


# Test continuous-time linear least squares with archiving
srand(1)
linear_sys = LinearSystem(-eye(2), 0.001*eye(2))
linear_obs = LinearObserver(eye(2), 0.01*eye(2))
initial_est = UncertainContinuousState([1.0, 2.0], 0.1*eye(2))
lse = LeastSquaresEstimator(linear_sys, linear_obs, initial_est)
simulator = make_simulator(lse)
for idx = 1:10
    true_state, measurement = simulate(simulator, (idx-1)*0.1)
    add!(lse, measurement)
end
archive = EstimatorHistory(lse)
@test distance(solve(lse, archive), initial_est) < 0.2
@test (length(archive.states) == 10) & (length(archive.residuals) == 10)


# Test discrete-time linear least squares
srand(1)
linear_sys = LinearSystem(eye(3), 0.001*eye(3))
linear_obs = LinearObserver(eye(3), 0.01*eye(3))
initial_est = UncertainDiscreteState([1.0, 2.0, 3.0], 0.1*eye(3))
lse = LeastSquaresEstimator(linear_sys, linear_obs, initial_est)
simulator = make_simulator(lse)
measurements = []
for idx = 1:10
    true_state, measurement = simulate(simulator, idx-1)
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
lse = LeastSquaresEstimator(linear_sys, linear_obs, initial_est)
simulator = make_simulator(lse)
measurements = []
for idx = 1:10
    true_state, measurement = simulate(simulator, idx-1)
    push!(measurements, measurement)
end
add!(lse, measurements)
archive = EstimatorHistory(lse)
solve!(lse, archive)
@test distance(lse.estimate, initial_est) < 0.2
@test (length(archive.states) == 10) & (length(archive.residuals) == 10)



# Test continuous-time nonlinear least squares
discrete_nl_fcn(t, x::Vector) = -x;
discrete_nl_jac(t, x::Vector) = -eye(length(x))
nonlin_sys = NonlinearSystem(discrete_nl_fcn, discrete_nl_jac, 0.000001*eye(2))
discrete_nl_fcn(t, x::Vector) = x;
discrete_nl_jac(t, x::Vector) = eye(length(x))
nonlin_obs = NonlinearObserver(discrete_nl_fcn, discrete_nl_jac, 0.0*eye(2))
initial_est = UncertainContinuousState([1.0, 3.0], eye(2))
nlse = NonlinearLeastSquaresEstimator(nonlin_sys, nonlin_obs, initial_est)
simulator = make_simulator(nlse)
for idx = 1:10
    true_state, measurement = simulate(simulator, idx*0.1)
    add!(nlse, measurement)
    if idx == 1
        println(true_state)
    end
end
solve!(nlse)
println(nlse.estimate)
