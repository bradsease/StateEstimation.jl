# Extended Kalman Filter tests


# System setup
discrete_nl_fcn(t, x::Vector) = x;
discrete_nl_jac(t, x::Vector) = eye(length(x))
nonlin_sys = NonlinearSystem(discrete_nl_fcn, discrete_nl_jac, eye(2))
nonlin_obs = NonlinearObserver(discrete_nl_fcn, discrete_nl_jac, eye(2))
initial_est = UncertainDiscreteState([1.0, 2.0], eye(2))

# Test constructors
ExtendedKalmanFilter(nonlin_sys, nonlin_obs, initial_est)
ExtendedKalmanFilter(nonlin_sys, nonlin_obs, make_absolute(initial_est))
ExtendedKalmanFilter(nonlin_sys, nonlin_obs, initial_est, [2])
@test_throws DimensionMismatch ExtendedKalmanFilter(
    nonlin_sys, nonlin_obs, initial_est, [3])
@test_throws ArgumentError ExtendedKalmanFilter(
    nonlin_sys, nonlin_obs, initial_est, [3,3])


# Test discrete EKF
srand(1)
discrete_nl_fcn(t, x::Vector) = x;
discrete_nl_jac(t, x::Vector) = eye(length(x))
nonlin_sys = NonlinearSystem(discrete_nl_fcn, discrete_nl_jac, eye(2))
nonlin_obs = NonlinearObserver(discrete_nl_fcn, discrete_nl_jac, eye(2))
initial_est = UncertainDiscreteState([1.0, 2.0], eye(2))
ekf = ExtendedKalmanFilter(nonlin_sys, nonlin_obs, initial_est)
simulator = make_simulator(ekf)
for i = 1:10
    true_state, measurement = simulate(simulator, i)
    process!(ekf, measurement)
    @test mahalanobis(true_state, ekf.estimate) < 3
end

# Test continuous EKF
srand(1)
F(t,x) = [x[2], -sin(x[1])]
dF_dx(t,x) = [0 1; -cos(x[1]) 0]
nonlinear_sys = NonlinearSystem(F, dF_dx, (0.01*pi/180)*eye(2,2))
linear_obs = LinearObserver([1.0, 0.0]', 0.1*pi/180)
initial_est = UncertainContinuousState([pi/4, 0.0], diagm([0.1, 0.01])*pi/180)
ekf = ExtendedKalmanFilter(nonlinear_sys, linear_obs, initial_est);
simulator = make_simulator(ekf)
for t = 0.1:0.1:10
    true_state, measurement = simulate(simulator, t)
    process!(ekf, measurement)
    @test mahalanobis(true_state, ekf.estimate) < 3
end


# Test convenience function
srand(1)
discrete_nl_fcn(t, x::Vector) = x;
discrete_nl_jac(t, x::Vector) = eye(length(x))
nonlin_sys = NonlinearSystem(discrete_nl_fcn, discrete_nl_jac, eye(2))
nonlin_obs = NonlinearObserver(discrete_nl_fcn, discrete_nl_jac, eye(2))
estimate = UncertainDiscreteState([1.0, 2.0], eye(2))
simulator =
    make_simulator(ExtendedKalmanFilter(nonlin_sys, nonlin_obs, estimate))
for i = 1:5
    true_state, measurement = simulate(simulator, i)
    estimate =
        extended_kalman_filter(estimate, nonlin_sys, nonlin_obs, measurement)
end

# Test convenience function with archiving
simulator =
    make_simulator(ExtendedKalmanFilter(nonlin_sys, nonlin_obs, estimate))
archive =
    EstimatorHistory(ExtendedKalmanFilter(nonlin_sys, nonlin_obs, estimate))
for i = 1:5
    true_state, measurement = simulate(simulator, i)
    estimate = extended_kalman_filter(
        estimate, nonlin_sys, nonlin_obs, measurement, archive)
end
