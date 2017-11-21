using StateEstimation
using Base.Test

# Sanity check
@test 1 == 1


# Test constructors
@test size(LinearSystem(1.0).A) == (1,1)
@test size(LinearSystem(1.0, 1.0).Q) == (1,1)
@test size(LinearSystem(eye(3), eye(3)).A) == (3,3)

#
lin_sys = LinearSystem(ones(2,2), eye(2))
@test state_transition_matrix(lin_sys, DiscreteState([2.0]), 10) == ones(2,2)^10
lin_sys = LinearSystem(2.0, 1.0)

# Test consistency between predict and state_transition_matrix
predict(lin_sys, DiscreteState(2.0), 2).x ==
    state_transition_matrix(lin_sys, DiscreteState(2.0), 2) * 2.0
predict(lin_sys, UncertainDiscreteState(2.0, 1.0), 2).x ==
    state_transition_matrix(lin_sys, UncertainDiscreteState(2.0, 1.0), 2) * 2.0

# Test prediction methods
lin_sys = LinearSystem(ones(2,2), eye(2))
@test predict(lin_sys, DiscreteState(ones(2)), 2) == DiscreteState(4*ones(2), 2)
@test predict(lin_sys, UncertainDiscreteState(ones(2), eye(2)), 2) ==
    UncertainDiscreteState(4*ones(2), eye(2)+10*ones(2,2), 2)

@test predict(lin_sys, ContinuousState(ones(2)), 2.0) ==
    ContinuousState(expm(lin_sys.A*2)*ones(2), 2.0)
predict(lin_sys, UncertainDiscreteState(ones(2), eye(2)), 2)

# Create linear system
#linear_sys = LinearSystem(eye(2))
#@test linear_sys.Q == zeros(2, 2)
#linear_sys = LinearSystem(eye(2), eye(2))

# Predict discrete state through linear system
#discrete_state = DiscreteState(ones(2))
#@inferred predict(linear_sys, discrete_state, 2)
#result = predict(linear_sys, discrete_state, 2)
#@test result.x == discrete_state.x

# Predict uncertain discrete state through linear system
#unc_discrete_state = UncertainDiscreteState(ones(2), eye(2))
#@inferred predict(linear_sys, unc_discrete_state, 2)
#result = predict(linear_sys, unc_discrete_state, 2)
#println(unc_discrete_state.P)
#println(unc_discrete_state)
#@test result.x == unc_discrete_state.x
#@test result.P == unc_discrete_state.P+linear_sys.Q

#
#@inferred simulate(linear_sys, unc_discrete_state)
#@inferred simulate(linear_sys, unc_continuous_state)
