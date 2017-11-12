using StateEstimation
using Base.Test

# Sanity check
@test 1 == 1


# Create discrete states
discrete_state = DiscreteState(ones(2))
discrete_state = DiscreteState(ones(2), 0)

# Create uncertain discrete states
unc_discrete_state = UncertainDiscreteState(ones(2), eye(2))
unc_discrete_state = UncertainDiscreteState(ones(2), eye(2), 0)

# Create continuous states
continuous_state = ContinuousState(ones(2))
continuous_state = ContinuousState(ones(2), 0.0)

# Create uncertain continuous tates
unc_continuous_state = UncertainContinuousState(ones(2), eye(2))
unc_continuous_state = UncertainContinuousState(ones(2),
                                                               eye(2), 0.0)

# Convert state types with constructors
@inferred DiscreteState(unc_discrete_state)
@inferred UncertainDiscreteState(discrete_state)
@inferred UncertainDiscreteState(discrete_state, eye(2))
@inferred ContinuousState(unc_continuous_state)
@inferred UncertainContinuousState(continuous_state)
@inferred UncertainContinuousState(continuous_state, eye(2))

# Convert state types with general methods
@inferred make_absolute(unc_continuous_state)
@inferred make_absolute(unc_discrete_state)
@inferred make_uncertain(continuous_state)
@inferred make_uncertain(continuous_state, eye(2))
@inferred make_uncertain(discrete_state)
@inferred make_uncertain(discrete_state, eye(2))

# Sample uncertain states
@inferred sample(unc_discrete_state)
@inferred sample(unc_continuous_state)
