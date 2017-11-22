using StateEstimation
using Base.Test

# Sanity check
@test 1 == 1


# Test constructors
@test size(LinearSystem(ones(2,2)).A) == (2,2)
@test size(LinearSystem(1.0).A) == (1,1)
@test size(LinearSystem(1.0, 1.0).Q) == (1,1)
@test size(LinearSystem(eye(3), eye(3)).A) == (3,3)

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
lin_sys = LinearSystem([[0.0, 1.0] [-1.0, 0.0]], 2*eye(2))
state = ContinuousState(ones(2))
predict!(lin_sys, state, 2.0)
@test state == ContinuousState(expm(lin_sys.A*2)*ones(2), 2.0)
state = UncertainContinuousState(ones(2), eye(2))
predict!(lin_sys, state, 2.0)
@test state.x == expm(lin_sys.A*2)*ones(2)
@test isapprox(state.P, [[5.0, 0.0] [0.0, 5.0]], rtol=1e-6)



#
#@inferred simulate(linear_sys, unc_discrete_state)
#@inferred simulate(linear_sys, unc_continuous_state)
