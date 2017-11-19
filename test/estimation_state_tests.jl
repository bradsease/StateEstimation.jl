using StateEstimation
using Base.Test

# Sanity check
@test 1 == 1



function test_state_addition_subtraction(state::AbstractState)
    for a = -1:0.5:1
        @test (state + a).x == (state.x+a)
        @test (state - a).x == (state.x-a)
        new_state = deepcopy(state)
        new_state .-= a
        @test new_state.x == (state.x-a)
        new_state = deepcopy(state)
        new_state .+= a
        @test new_state.x == (state.x+a)
    end
end
test_state_addition_subtraction(DiscreteState(ones(2)))
test_state_addition_subtraction(ContinuousState(ones(2)))
test_state_addition_subtraction(UncertainDiscreteState(ones(2), eye(2)))
test_state_addition_subtraction(UncertainContinuousState(ones(2), eye(2)))

function test_state_scalar_multiplication(state::AbstractState)
    for a = -1:0.5:1
        @test (state*a).x == (state.x*a)
        @test (a*state).x == (state.x*a)
        new_state = deepcopy(state)
        new_state .*= a
        @test new_state.x == (state.x*a)
        if typeof(state) <: AbstractUncertainState
            @test (state*a).P == (state.P*a^2)
            @test (a*state).P == (state.P*a^2)
            new_state = deepcopy(state)
            new_state .*= a
            @test new_state.P == (state.P*a^2)
        end
    end
end
test_state_scalar_multiplication(DiscreteState(ones(2)))
test_state_scalar_multiplication(ContinuousState(ones(2)))
test_state_scalar_multiplication(UncertainDiscreteState(ones(2), eye(2)))
test_state_scalar_multiplication(UncertainContinuousState(ones(2), eye(2)))

# State equality checks
@test DiscreteState(ones(2)) == DiscreteState([1.0, 1.0])
@test UncertainDiscreteState(ones(2), eye(2)) ==
    UncertainDiscreteState([1.0, 1.0], eye(2))
@test UncertainDiscreteState(ones(2), eye(2)) != DiscreteState(ones(2))
@test DiscreteState(ones(2)) != ContinuousState(ones(2))
@test ContinuousState(ones(2)) == ContinuousState([1.0, 1.0])
@test UncertainContinuousState(ones(2), eye(2)) ==
    UncertainContinuousState([1.0, 1.0], eye(2))
@test UncertainContinuousState(ones(2), eye(2)) != ContinuousState(ones(2))

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
unc_continuous_state = UncertainContinuousState(ones(2), eye(2), 0.0)

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

# Test distance metrics
@test distance(discrete_state, discrete_state) == 0
@test distance(discrete_state, unc_discrete_state) == 0
@test mahalanobis(discrete_state, unc_continuous_state) == 0
@test mahalanobis(DiscreteState([4.0, 1.0]), unc_discrete_state) == 3
@test mahalanobis(discrete_state, unc_discrete_state) ==
    mahalanobis(discrete_state, discrete_state, unc_discrete_state.P)
@inferred distance(discrete_state, discrete_state)
@inferred mahalanobis(discrete_state, unc_continuous_state)
