# Observer tests


# Test constructors
@test size(LinearObserver(ones(2,2)).H) == (2,2)
@test size(LinearObserver(1.0).H) == (1,1)
@test size(LinearObserver(1.0, 1.0).R) == (1,1)
@test size(LinearObserver(eye(3), eye(3)).H) == (3,3)
@test size(LinearObserver(ones(3), eye(3)).H) == (3,1)
@test size(LinearObserver(ones(3)', 1.0).R) == (1,1)
@test size(LinearObserver(ones(3)', ones(1,1)').R) == (1,1)

# Test compatibility methods
lin_obs = LinearObserver(ones(2,2), eye(2))
StateEstimation.assert_compatibility(lin_obs, DiscreteState(ones(2)))
StateEstimation.assert_compatibility(DiscreteState(ones(2)), lin_obs)
@test_throws DimensionMismatch StateEstimation.assert_compatibility(lin_obs,
    DiscreteState(ones(3)))
lin_obs = LinearObserver(ones(2)', 2.0)
StateEstimation.assert_compatibility(DiscreteState([1.0]), lin_obs)
@test_throws DimensionMismatch StateEstimation.assert_compatibility(
    DiscreteState(ones(2)), lin_obs)

# Test discrete prediction methods
lin_obs = LinearObserver(ones(2,2), eye(2))
@test predict(lin_obs, DiscreteState(ones(2))) == DiscreteState(2*ones(2), 0)
@test predict(lin_obs, UncertainDiscreteState(ones(2), eye(2))) ==
    UncertainDiscreteState(2*ones(2), lin_obs.H*lin_obs.H'+lin_obs.R, 0)

# Test continuous prediction methods
lin_obs = LinearObserver(ones(2)', 1.0)
@test predict(lin_obs, ContinuousState(ones(2))) == ContinuousState(2.0, 0.0)
@test predict(lin_obs, UncertainContinuousState(ones(2), eye(2))) ==
    UncertainContinuousState(2.0*ones(1), lin_obs.H*lin_obs.H'+lin_obs.R, 0.0)

# Test discrete measure methods
lin_obs = LinearObserver(ones(2,2), eye(2))
@test measure(lin_obs, DiscreteState(ones(2))) == DiscreteState(2*ones(2), 0)
@test measure(lin_obs, UncertainDiscreteState(ones(2), eye(2))) ==
    DiscreteState(2*ones(2), 0)

# Test continuous measure methods
lin_obs = LinearObserver(ones(2)', 1.0)
@test measure(lin_obs, ContinuousState(ones(2))) == ContinuousState(2.0, 0.0)
@test measure(lin_obs, UncertainContinuousState(ones(2), eye(2))) ==
    ContinuousState(2.0*ones(1), 0.0)

# Test observability methods
linear_sys = LinearSystem(Float64[[1, -2] [-3, -4]]', 0.1*eye(2))
linear_obs = LinearObserver(reshape(Float64[1, 2], 1, 2), reshape([0.1], 1, 1))
@test observable(linear_sys, linear_obs) == false
@test observable(eye(2), eye(2))



# Test nonlinear constructors
discrete_nl_fcn(t, x::Vector) = x.^2;
discrete_nl_jac(t, x::Vector) = diagm(2*x)
NonlinearObserver(discrete_nl_fcn, discrete_nl_jac, eye(2))
NonlinearObserver(discrete_nl_fcn, discrete_nl_jac, 1.0)

# Test nonlinear absolute state prediction methods
nonlin_obs = NonlinearObserver(discrete_nl_fcn, discrete_nl_jac, eye(3))
@test predict(nonlin_obs, DiscreteState([0.0, 1.0, 2.0])) ==
    DiscreteState([0.0, 1.0, 4.0], 0)
@test predict(nonlin_obs, ContinuousState([0.0, 2.0, 4.0], 1.0)) ==
    ContinuousState([0.0, 4.0, 16.0], 1.0)

# Test nonlinear uncertain state prediction methods
@test predict(nonlin_obs, UncertainDiscreteState([0.0, 1.0, 2.0], eye(3))) ==
    UncertainDiscreteState([0.0, 1.0, 4.0], diagm([1.0, 5.0, 17.0]), 0)
@test predict(nonlin_obs, UncertainContinuousState([0.0, 1.0, 2.0], eye(3))) ==
    UncertainContinuousState([0.0, 1.0, 4.0], diagm([1.0, 5.0, 17.0]), 0.0)

# Test nonlinear discrete measure methods
@test measure(nonlin_obs, DiscreteState([0.0, 1.0, 2.0])) ==
    DiscreteState([0.0, 1.0, 4.0], 0)
@test measure(nonlin_obs, UncertainDiscreteState([0.0, 1.0, 2.0], eye(3))) ==
    measure(nonlin_obs, DiscreteState([0.0, 1.0, 2.0]))

# Test nonlinear continuous measure methods
@test measure(nonlin_obs, ContinuousState([0.0, 1.0, 2.0])) ==
    ContinuousState([0.0, 1.0, 4.0], 0.0)
@test measure(nonlin_obs, UncertainContinuousState([0.0, 1.0, 2.0], eye(3))) ==
    measure(nonlin_obs, ContinuousState([0.0, 1.0, 2.0]))
