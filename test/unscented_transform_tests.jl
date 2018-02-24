# Unscented Transform


# Test constructors
ut_params = UnscentedTransform()
@test ut_params == UnscentedTransform(1e-3, 2, 0)

# Test linear system augment methods
sys = LinearSystem(eye(2), 2.25*eye(2))
dstate = UncertainDiscreteState(1.5*ones(2), 3.0*eye(2))
astate, asys = augment(dstate, sys)
@test (length(astate.x) == 4) & (size(astate.P) == (4,4)) & (size(asys.A) == (4,4))

# Test consistency between linear prediction and UnscentedTransform
nominal_prediction = predict(dstate, sys, 1)
astate, asys = augment(dstate, sys)
disc_fcn(x) = predict(x, asys, 1)
unscented_prediction = transform(astate, disc_fcn, ut_params)
@test isapprox(nominal_prediction.x, unscented_prediction.x[1:2])
@test isapprox(nominal_prediction.P, unscented_prediction.P[1:2,1:2])

# Test nonlinear system augment methods
F(t,x) = x.^2
dF(t,x) = diagm(2*x)
nsys = NonlinearSystem(F, dF, 2*eye(2))
dstate = UncertainDiscreteState(1.5*ones(2), 3.0*eye(2))
astate, asys = augment(dstate, nsys)
@test length(astate.x) == 4

# Test linear observer augment methods
obs = LinearObserver([1.0, 2.0]', 1.0)
dstate = UncertainDiscreteState(1.5*ones(2), 3.0*eye(2))
astate, aobs = augment(dstate, obs)
@test (length(astate.x) == 3) & (size(astate.P) == (3,3)) & (size(aobs.H) == (1,3))

# Test nonlinear observer augment methods
H(t,x) = [x[1].^2]
dH(t,x) = [2.0*x[1] 0.0]
nobs = NonlinearObserver(H, dH, 2.0)
dstate = UncertainDiscreteState(1.5*ones(2), 3.0*eye(2))
astate, aobs = augment(dstate, nobs)
@test length(astate.x) == 3


# TODO: Test combined augment methods
augment(dstate, sys, obs)
augment(dstate, nsys, obs)
augment(dstate, sys, nobs)
augment(dstate, nsys, nobs)

# Test constructors
UnscentedKalmanFilter(nsys, nobs, dstate)
UnscentedKalmanFilter(nsys, nobs, dstate, UnscentedTransform())
UnscentedKalmanFilter(nsys, nobs, dstate, [1])


# Test linear observer prediction
obs = LinearObserver(eye(2), 2*eye(2))
ut_state = predict(dstate, obs, ut_params)
expected_state = predict(dstate, obs)
@test isapprox(ut_state.x, expected_state.x)
@test isapprox(ut_state.P, expected_state.P)

# Test linear system prediction
sys = LinearSystem(eye(2), 2*eye(2))
ut_state = predict(dstate, sys, ut_params, 2)
expected_state = predict(dstate, sys, 2)
@test isapprox(ut_state.x, expected_state.x)
@test isapprox(ut_state.P, expected_state.P)


# Test linear ukf
srand(1);
linear_sys = LinearSystem(0.5*eye(2), 0.001*eye(2))
linear_obs = LinearObserver(eye(2), 0.001*eye(2))
initial_est = UncertainDiscreteState([1.0, 2.0], 0.1*eye(2))
ukf = UnscentedKalmanFilter(linear_sys, linear_obs, initial_est)
kf = KalmanFilter(linear_sys, linear_obs, initial_est)
simulator = make_simulator(kf)
for i = 1:4
    true_state, measurement = simulate(simulator, i)
    process!(ukf, measurement)
    process!(kf, measurement)
    @test isapprox(ukf.estimate.x, kf.estimate.x)
end

srand(1);
linear_sys = LinearSystem(0.5*eye(2), 0.001*eye(2))
linear_obs = LinearObserver(eye(2), 0.001*eye(2))
initial_est = UncertainDiscreteState([1.0, 2.0], 0.1*eye(2))
ukf = UnscentedKalmanFilter(
    NonlinearSystem(linear_sys), NonlinearObserver(linear_obs), initial_est)
kf = KalmanFilter(linear_sys, linear_obs, initial_est)
simulator = make_simulator(kf)
for i = 1:4
    true_state, measurement = simulate(simulator, i)
    process!(ukf, measurement)
    process!(kf, measurement)
    @test isapprox(ukf.estimate.x, kf.estimate.x)
end

# Test nonlinear UKF with linear observer
srand(1);
F(t,x) = [sin(x[2]), -x[1]]
dF_dx(t,x) = [0 cos(x[2]); -1 0]
nonlinear_sys = NonlinearSystem(F, dF_dx, 0.01*eye(2,2))
linear_obs = LinearObserver(eye(2), 0.1*eye(2))
initial_est = UncertainDiscreteState([0.0, 2.0], 0.1*eye(2))
ukf = UnscentedKalmanFilter(nonlinear_sys, linear_obs, initial_est);
simulator = make_simulator(ukf)
for idx = 1:10
    true_state, measurement = simulate(simulator, idx)
    process!(ukf, measurement)
    @test mahalanobis(true_state, ukf.estimate) < 3
end




# Test convenience function
srand(1);
F(t,x) = [sin(x[2]), -x[1]]
dF_dx(t,x) = [0 cos(x[2]); -1 0]
nonlinear_sys = NonlinearSystem(F, dF_dx, 0.01*eye(2,2))
linear_obs = LinearObserver(eye(2), 0.1*eye(2))
estimate = UncertainDiscreteState([0.0, 2.0], 0.1*eye(2))
ut_params = UnscentedTransform()
ukf = UnscentedKalmanFilter(nonlinear_sys, linear_obs, estimate)
simulator = make_simulator(ukf)
for idx = 1:10
    true_state, measurement = simulate(simulator, idx)
    estimate = unscented_kalman_filter(
        estimate, nonlinear_sys, linear_obs, ut_params, measurement)
end

# Test convenience function with archiving
simulator = make_simulator(ukf)
archive = EstimatorHistory(ukf)
for idx = 1:10
    true_state, measurement = simulate(simulator, idx)
    estimate = unscented_kalman_filter(
        estimate, nonlinear_sys, linear_obs, ut_params, measurement, archive)
end
