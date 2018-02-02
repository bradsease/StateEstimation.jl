# Unscented Transform


# Test constructors
@test UnscentedTransform() == UnscentedTransform(1e-3, 2, 0)

# Test methods
state = UncertainContinuousState(ones(3), diagm([1.0, 2.0, 3.0]))
ut = UnscentedTransform()
compute_weights(ut, 3)
compute_sigma_points(state, ut)


#
state = UncertainContinuousState(1.0, 0.1)
sys = LinearSystem(2.0)
# predict(state, sys, ut, 1.0)


# Test linear system augment methods
sys = LinearSystem(eye(2), 2.25*eye(2))
dstate = UncertainDiscreteState(1.5*ones(2), 3.0*eye(2))
astate, asys = augment(dstate, sys)
@test (length(astate.x) == 4) & (size(astate.P) == (4,4)) & (size(asys.A) == (4,4))


# Test consistency between linear prediction and UnscentedTransform
nominal_prediction = predict(dstate, sys, 1)
astate, asys = augment(dstate, sys)
disc_fcn(x) = predict(x, asys, 1)
unscented_prediction = transform(astate, disc_fcn, ut)
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
UnscentedKalmanFilter(nsys, nobs, dstate)
UnscentedKalmanFilter(nsys, nobs, dstate, UnscentedTransform())
UnscentedKalmanFilter(nsys, nobs, dstate, [1])




# Test linear observer prediction
obs = LinearObserver(eye(2), 2*eye(2))
ut_state = predict(dstate, obs, ut)
expected_state = predict(dstate, obs)
@test isapprox(ut_state.x, expected_state.x)
@test isapprox(ut_state.P, expected_state.P)

# Test linear system prediction
sys = LinearSystem(eye(2), 2*eye(2))
ut_state = predict(dstate, sys, ut, 2)
expected_state = predict(dstate, sys, 2)
@test isapprox(ut_state.x, expected_state.x)
@test isapprox(ut_state.P, expected_state.P)
