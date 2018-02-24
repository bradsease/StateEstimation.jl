# Archive tests

#
state_history = [UncertainDiscreteState(1.0*idx, 1.0) for idx in 1:10]
times, states, sigmas =
    StateEstimation.vectorize_state_history(state_history)
@test length(times) == 10
