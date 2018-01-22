# Unscented Transform


# Test constructors
@test UnscentedTransform() == UnscentedTransform(1e-3, 2, 0)

# Test methods
state = UncertainContinuousState(ones(3), diagm([1.0, 2.0, 3.0]))
transform = UnscentedTransform()
compute_weights(transform, 3)
compute_sigma_points(state, transform)


#
state = UncertainContinuousState(1.0, 0.1)
sys = LinearSystem(2.0)
predict(state, sys, transform, 1.0)
#println(predict(state, sys, transform, 1.0))
#println(predict(state, sys, 1.0))


# Test linear system augment methods
sys = LinearSystem(eye(2), 2.25*eye(2))
cstate = UncertainContinuousState(1.5*ones(2), 3.0*eye(2))
dstate = UncertainDiscreteState(1.5*ones(2), 3.0*eye(2))
astate, asys = augment(cstate, sys)
@test (length(astate.x) == 4) & (size(astate.P) == (4,4)) & (size(asys.A) == (4,4))
astate, asys = augment(dstate, sys)
@test (length(astate.x) == 4) & (size(astate.P) == (4,4)) & (size(asys.A) == (4,4))

# Test nonlinear system augment methods
F(t,x) = x.^2
dF(t,x) = diagm(2*x)
nsys = NonlinearSystem(F, dF, 2*eye(2))
cstate = UncertainContinuousState(1.5*ones(2), 3.0*eye(2))
dstate = UncertainDiscreteState(1.5*ones(2), 3.0*eye(2))
astate, asys = augment(cstate, nsys)
@test length(astate.x) == 4
astate, asys = augment(dstate, nsys)
@test length(astate.x) == 4

# Test linear observer augment methods
obs = LinearObserver([1.0, 2.0]', 1.0)
cstate = UncertainContinuousState(1.5*ones(2), 3.0*eye(2))
dstate = UncertainDiscreteState(1.5*ones(2), 3.0*eye(2))
astate, aobs = augment(cstate, obs)
@test (length(astate.x) == 3) & (size(astate.P) == (3,3)) & (size(aobs.H) == (1,3))
astate, asys = augment(dstate, obs)
@test (length(astate.x) == 3) & (size(astate.P) == (3,3)) & (size(aobs.H) == (1,3))

# Test nonlinear observer augment methods
H(t,x) = [x[1].^2]
dH(t,x) = [2.0*x[1] 0.0]
nobs = NonlinearObserver(H, dH, 2.0)
cstate = UncertainContinuousState(1.5*ones(2), 3.0*eye(2))
dstate = UncertainDiscreteState(1.5*ones(2), 3.0*eye(2))
astate, aobs = augment(cstate, nobs)
@test length(astate.x) == 3
astate, aobs = augment(dstate, nobs)
@test length(astate.x) == 3


# TODO: Test combined augment methods
augment(cstate, sys, obs)
augment(dstate, sys, obs)

augment(cstate, nsys, obs)
augment(dstate, nsys, obs)

augment(cstate, sys, nobs)
augment(dstate, sys, nobs)

augment(cstate, nsys, nobs)
augment(dstate, nsys, nobs)
