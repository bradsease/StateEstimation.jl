# Sytem tests


# Test constructors
@test size(LinearSystem(ones(2,2)).A) == (2,2)
@test size(LinearSystem(1.0).A) == (1,1)
@test size(LinearSystem(1.0, 1.0).Q) == (1,1)
@test size(LinearSystem(eye(3), eye(3)).A) == (3,3)
@test_throws DimensionMismatch LinearSystem(eye(2), eye(3))
@test_throws DimensionMismatch LinearSystem(eye(2,3), eye(2))
@test_throws DimensionMismatch predict(LinearSystem(1.0),
                                       DiscreteState(ones(2)), 1)

# Test state transtion matrix methods
lin_sys = LinearSystem(ones(2,2), eye(2))
state = DiscreteState(2*ones(2))
@test state_transition_matrix(lin_sys, state, 10) == ones(2,2)^10
lin_sys = LinearSystem(2.0, 1.0)
state = ContinuousState([1.0])
@test state_transition_matrix(lin_sys, state, 2.0) == exp(4.0)*ones(1,1)

# Test consistency between predict and state_transition_matrix
predict(lin_sys, DiscreteState(2.0), 2).x ==
    state_transition_matrix(lin_sys, DiscreteState(2.0), 2) * 2.0
predict(lin_sys, UncertainDiscreteState(2.0, 1.0), 2).x ==
    state_transition_matrix(lin_sys, UncertainDiscreteState(2.0, 1.0), 2) * 2.0

# Test discrete prediction methods
lin_sys = LinearSystem(ones(2,2), eye(2))
@test predict(lin_sys, DiscreteState(ones(2)), 2) == DiscreteState(4*ones(2), 2)
@test predict(lin_sys, UncertainDiscreteState(ones(2), eye(2)), 2) ==
    UncertainDiscreteState(4*ones(2), eye(2)+10*ones(2,2), 2)

# Test continuous prediction methods
lin_sys = LinearSystem([[0.0, 1.0] [-1.0, 0.0]], 2*eye(2))
@test predict(lin_sys, ContinuousState(ones(2)), 2.0) ==
    ContinuousState(expm(lin_sys.A*2)*ones(2), 2.0)
result = predict(lin_sys, UncertainContinuousState(ones(2), eye(2)), 2.0)
@test result.x == expm(lin_sys.A*2)*ones(2)
@test isapprox(result.P, [[5.0, 0.0] [0.0, 5.0]], rtol=1e-6)

# Test in-place discrete prediction methods
lin_sys = LinearSystem(ones(2,2), eye(2))
state = DiscreteState(ones(2))
predict!(lin_sys, state, 2)
@test state == DiscreteState(4*ones(2), 2)
state = UncertainDiscreteState(ones(2), eye(2))
predict!(lin_sys, state, 2)
@test state == UncertainDiscreteState(4*ones(2), eye(2)+10*ones(2,2), 2)

# Test in-place continuous prediction methods
lin_sys = LinearSystem([[0.0, 1.0] [-1.0, 1.0]], 2*eye(2))
state = ContinuousState(ones(2))
predict!(lin_sys, state, 0.1)
@test state == ContinuousState(expm(lin_sys.A*0.1)*ones(2), 0.1)
state = UncertainContinuousState(ones(2), eye(2))
predict!(lin_sys, state, 0.1)
@test state.x == expm(lin_sys.A*0.1)*ones(2)
@test isapprox(state.P, [[1.20075, -0.0117417] [-0.0117417, 1.44201]],rtol=1e-5)

# Test simulation methods
lin_sys = LinearSystem([[0.0, 1.0] [-1.0, 1.0]], 2*eye(2))
continuous_state = ContinuousState(ones(2))
uncertain_discrete_state = UncertainDiscreteState(ones(2), eye(2))
@test typeof(simulate(lin_sys, continuous_state, 1.0)) <: ContinuousState
@test typeof(simulate(lin_sys, uncertain_discrete_state, 1)) <: DiscreteState


# Test nonlinear constructors
discrete_nl_fcn(t, x::Vector) = x.^2;
discrete_nl_jac(t, x::Vector) = diagm(2*x)
NonlinearSystem(discrete_nl_fcn, discrete_nl_jac, 1.0)
NonlinearSystem(discrete_nl_fcn, discrete_nl_jac, eye(2))

# Test state transtion matrix methods
continuous_nl_fcn(t, x::Vector) = -x.^2;
continuous_nl_jac(t, x::Vector) = -diagm(2*x)
continuous_state = ContinuousState(ones(2))
discrete_state = DiscreteState(ones(2))
nonlin_sys = NonlinearSystem(continuous_nl_fcn, continuous_nl_jac, eye(2))
@test size(state_transition_matrix(nonlin_sys, continuous_state, 1.0)) == (2,2)
@test size(state_transition_matrix(nonlin_sys, discrete_state, 1)) == (2,2)

# Test nonlinear discrete prediction methods
nonlin_sys = NonlinearSystem(discrete_nl_fcn, discrete_nl_jac, eye(3))
@test predict(nonlin_sys, UncertainDiscreteState([0.0, 1.0, 2.0], eye(3)), 1) ==
    UncertainDiscreteState([0.0, 1.0, 4.0], diagm([1.0, 5.0, 65.0]), 1)
@test predict(nonlin_sys, DiscreteState([0.0, 1.0, 2.0]), 1) ==
    DiscreteState([0.0, 1.0, 4.0], 1)

# Test nonlinear continuous prediction methods
continuous_nl_fcn(t, x::Vector) = [x[2], 0.0]
continuous_nl_jac(t, x::Vector) = [0.0 1.0; 0.0 0.0]
nonlin_sys = NonlinearSystem(continuous_nl_fcn, continuous_nl_jac, eye(2))
@test isapprox(predict(nonlin_sys,  ContinuousState([0.0, 1.0]), 2).x,
    ContinuousState([2.0, 1.0], 2.0).x)

# Test consistency between linear and nonlinear prediction
lin_sys = LinearSystem([0.0 1.0; 0.0 0.0], eye(2))
@test isapprox(
    predict(nonlin_sys,  UncertainContinuousState([0.0, 1.0], eye(2)), 2.0).P,
    predict(lin_sys,  UncertainContinuousState([0.0, 1.0], eye(2)), 2.0).P)

# Test simulation methods
continuous_nl_fcn(t, x::Vector) = [x[2], 0.0]
continuous_nl_jac(t, x::Vector) = [0.0 1.0; 0.0 0.0]
nonlin_sys = NonlinearSystem(continuous_nl_fcn, continuous_nl_jac, eye(2))
@test typeof(simulate(nonlin_sys, ContinuousState([0.0, 1.0]), 1.0)) <:
    ContinuousState
