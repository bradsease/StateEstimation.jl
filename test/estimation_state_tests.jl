using StateEstimation
using Base.Test

# Sanity check
@test 1 == 1


# Discrete state constructors
@test size(DiscreteState(1.0).x) == (1,)
@test size(DiscreteState(1.0, 0).x) == (1,)
@test size(UncertainDiscreteState(1.0, 2.0).x) == (1,)
@test size(UncertainDiscreteState(1.0, 2.0, 0).P) == (1, 1)
@test DiscreteState(1.0) == DiscreteState(UncertainDiscreteState(1.0, 2.0))
@test UncertainDiscreteState(ones(2), eye(2)) ==
    UncertainDiscreteState(DiscreteState(ones(2)), eye(2))

# Continuous state constructors
@test size(ContinuousState(1.0).x) == (1,)
@test size(ContinuousState(1.0, 0.0).x) == (1,)
@test size(UncertainContinuousState(1.0, 2.0).x) == (1,)
@test size(UncertainContinuousState(1.0, 2.0, 0.0).P) == (1, 1)
@test ContinuousState(1.0) == ContinuousState(UncertainContinuousState(1.0,2.0))
@test UncertainContinuousState(ones(2), eye(2)) ==
    UncertainContinuousState(ContinuousState(ones(2)), eye(2))

# Create testing states
discrete_state = DiscreteState(ones(2))
discrete_state = DiscreteState(ones(2), 0)
unc_discrete_state = UncertainDiscreteState(ones(2), eye(2))
unc_discrete_state = UncertainDiscreteState(ones(2), eye(2), 0)
continuous_state = ContinuousState(ones(2))
continuous_state = ContinuousState(ones(2), 0.0)
unc_continuous_state = UncertainContinuousState(ones(2), eye(2))
unc_continuous_state = UncertainContinuousState(ones(2), eye(2), 0.0)

# Test type conversions
@test StateEstimation.absolute_type(unc_discrete_state) <: DiscreteState
@test StateEstimation.absolute_type(unc_continuous_state) <: ContinuousState
@test StateEstimation.uncertain_type(discrete_state) <: UncertainDiscreteState
@test StateEstimation.uncertain_type(continuous_state) <: UncertainContinuousState
@test discrete_state == make_absolute(unc_discrete_state)
@test unc_discrete_state == make_uncertain(discrete_state, unc_discrete_state.P)
@test continuous_state == make_absolute(unc_continuous_state)
@test unc_continuous_state == make_uncertain(continuous_state,
                                             unc_continuous_state.P)

# Sample uncertain states
@test typeof(sample(unc_discrete_state)) <: DiscreteState
@test typeof(sample(unc_continuous_state)) <: ContinuousState

# Test distance metrics
@test distance(discrete_state, discrete_state) == 0
@test distance(discrete_state, unc_discrete_state) == 0
@test mahalanobis(discrete_state, unc_continuous_state) == 0
@test mahalanobis(DiscreteState([4.0, 1.0]), unc_discrete_state) == 3
@test mahalanobis(discrete_state, unc_discrete_state) ==
    mahalanobis(discrete_state, discrete_state, unc_discrete_state.P)

function test_state_addition_subtraction(state::AbstractState)
    for a = -1:0.5:1
        @test (state + a).x == (state.x+a)
        @test (a + state).x == (state.x+a)
        @test (state - a).x == (state.x-a)
        @test (a - state).x == (state.x-a)
        new_state = deepcopy(state)
        new_state .-= a
        new_state .+= a
        new_state .= a .+ new_state
        new_state .= a .- new_state
        @test new_state.x == state.x
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
        new_state .= a .* new_state
        @test new_state.x == (state.x*a^2)
        if typeof(state) <: AbstractUncertainState
            @test (state*a).P == (state.P*a^2)
            @test (a*state).P == (state.P*a^2)
            new_state = deepcopy(state)
            new_state .*= a
            @test new_state.P == (state.P*a^2)
            new_state .= a .* new_state
            @test new_state.P == (state.P*a^4)
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
@test (UncertainDiscreteState(ones(2), eye(2))==DiscreteState(ones(2))) == false
@test (DiscreteState(ones(2)) == ContinuousState(ones(2))) == false
@test ContinuousState(ones(2)) == ContinuousState([1.0, 1.0])
@test UncertainContinuousState(ones(2), eye(2)) ==
    UncertainContinuousState([1.0, 1.0], eye(2))
@test UncertainContinuousState(ones(2), eye(2)) != ContinuousState(ones(2))
