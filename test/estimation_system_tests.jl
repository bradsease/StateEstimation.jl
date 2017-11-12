using StateEstimation
using Base.Test

# Sanity check
@test 1 == 1


# Create linear system
linear_sys = LinearSystem(eye(2))
@test linear_sys.Q == zeros(2, 2)
linear_sys = LinearSystem(eye(2), eye(2))

# Predict discrete state through linear system
discrete_state = DiscreteState(ones(2))
@inferred predict(linear_sys, discrete_state)
result = predict(linear_sys, discrete_state)
@test result.x == discrete_state.x

# Predict uncertain discrete state through linear system
unc_discrete_state = UncertainDiscreteState(ones(2), eye(2))
@inferred predict(linear_sys, unc_discrete_state)
result = predict(linear_sys, unc_discrete_state)
@test result.x == unc_discrete_state.x
@test result.P == unc_discrete_state.P+linear_sys.Q

#
#@inferred simulate(linear_sys, unc_discrete_state)
#@inferred simulate(linear_sys, unc_continuous_state)
