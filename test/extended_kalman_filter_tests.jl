# Extended Kalman Filter tests


# System setup
discrete_nl_fcn(t, x::Vector) = x;
discrete_nl_jac(t, x::Vector) = eye(size(x))
nonlin_sys = NonlinearSystem(discrete_nl_fcn, discrete_nl_jac, eye(2), 2)
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
