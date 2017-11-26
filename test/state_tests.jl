# State tests


# Discrete state constructors
@test size(DiscreteState(1.0).x) == (1,)
@test size(DiscreteState(1.0, 0).x) == (1,)
@test size(UncertainDiscreteState(1.0, 2.0).x) == (1,)
@test size(UncertainDiscreteState(1.0, 2.0, 0).P) == (1, 1)
@test DiscreteState(1.0) == DiscreteState(UncertainDiscreteState(1.0, 2.0))
@test UncertainDiscreteState(ones(2), eye(2)) ==
    UncertainDiscreteState(DiscreteState(ones(2)), eye(2))
@test UncertainDiscreteState(ones(2), zeros(2,2)) ==
    UncertainDiscreteState(DiscreteState(ones(2)))

# Continuous state constructors
@test size(ContinuousState(1.0).x) == (1,)
@test size(ContinuousState(1.0, 0.0).x) == (1,)
@test size(UncertainContinuousState(1.0, 2.0).x) == (1,)
@test size(UncertainContinuousState(1.0, 2.0, 0.0).P) == (1, 1)
@test ContinuousState(1.0) == ContinuousState(UncertainContinuousState(1.0,2.0))
@test UncertainContinuousState(ones(2), eye(2)) ==
    UncertainContinuousState(ContinuousState(ones(2)), eye(2))
@test UncertainContinuousState(ones(2), zeros(2,2)) ==
    UncertainContinuousState(ContinuousState(ones(2)))

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
@test make_uncertain(discrete_state, zeros(2,2)) ==
    make_uncertain(discrete_state)
@test continuous_state == make_absolute(unc_continuous_state)
@test unc_continuous_state == make_uncertain(continuous_state,
                                             unc_continuous_state.P)
@test make_uncertain(continuous_state, zeros(2,2)) ==
    make_uncertain(continuous_state)

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

function test_state_matrix_multiplication(state::AbstractState)
    test_matrices = [eye(2), ones(2,2), -ones(2,2), zeros(2,2)]
    for idx = 1:length(test_matrices)
        A = test_matrices[idx]
        @test (A*state).x == A*state.x
        new_state = deepcopy(state)
        new_state .= A * state
        @test new_state.x == A*state.x
        if typeof(state) <: AbstractUncertainState
            @test (A*state).P == A*state.P*A'
            new_state = deepcopy(state)
            new_state .= A * state
            @test new_state.P == A*state.P*A'
        end
    end
end
test_state_scalar_multiplication(DiscreteState(ones(2)))
test_state_scalar_multiplication(ContinuousState(ones(2)))
test_state_scalar_multiplication(UncertainDiscreteState(ones(2), eye(2)))
test_state_scalar_multiplication(UncertainContinuousState(ones(2), eye(2)))

# Test in-place assignment operations
state1 = DiscreteState(ones(2))
state1 .= DiscreteState(zeros(2))
@test state1.x == zeros(2)
state1 = ContinuousState(ones(2))
state1 .= ContinuousState(zeros(2))
@test state1.x == zeros(2)
state1 = UncertainDiscreteState(ones(2), eye(2))
state1 .= UncertainDiscreteState(zeros(2), zeros(2,2))
@test (state1.x == zeros(2)) & (state1.P == zeros(2,2))
state1 = UncertainContinuousState(ones(2), eye(2))
state1 .= UncertainContinuousState(zeros(2), zeros(2,2))
@test (state1.x == zeros(2)) & (state1.P == zeros(2,2))

# State equality checks
@test DiscreteState(ones(2)) == DiscreteState([1.0, 1.0])
@test DiscreteState(ones(2)) != DiscreteState(zeros(2))
@test UncertainDiscreteState(ones(2), eye(2)) ==
    UncertainDiscreteState([1.0, 1.0], eye(2))
@test UncertainDiscreteState(ones(2), eye(2)) != DiscreteState(ones(2))
@test (DiscreteState(ones(2)) == ContinuousState(ones(2))) == false
@test ContinuousState(ones(2)) == ContinuousState([1.0, 1.0])
@test ContinuousState(ones(2)) != ContinuousState(zeros(2))
@test UncertainContinuousState(ones(2), eye(2)) ==
    UncertainContinuousState([1.0, 1.0], eye(2))
@test UncertainContinuousState(ones(2), eye(2)) != ContinuousState(ones(2))
