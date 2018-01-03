#
# Convenience simulators for estimator types
#


"""
    SingleStateSimulator(sys::AbstractSystem, obs::AbstractObserver,
                         state::AbstractUncertainState)

Single-state simulator type that mimics the structure of `Estimator` types. An
instance of SingleStateSimulator can be used to simulate a sequence of truth
states and measurements in the presence of system and observer noise.

In general, the best way to create a simulator is through the `make_simulator`
method.
"""
immutable SingleStateSimulator{T,S<:AbstractAbsoluteState{T}}
    sys::AbstractSystem{T}
    obs::AbstractObserver{T}
    true_state::S
end
function SingleStateSimulator(sys::AbstractSystem, obs::AbstractObserver,
                              state::AbstractUncertainState)
    return SingleStateSimulator(sys, obs, sample(state))
end


"""
    MultiStateSimulator(simulator_bank::Vector{SingleStateSimulator{T,S}})

Multi-state simulator type that contains a bank of `SingleStateSimulator`. This
simulator type mimics the structure of a `MultiTargetFilter` and can be used to
simulate a sequence of truth states and measurements in the presence of system
and observer noise for a collection of systems.

In general, the best way to create a simulator is through the `make_simulator`
method.
"""
immutable MultiStateSimulator{T,S<:AbstractAbsoluteState{T}}
    simulator_bank::Vector{SingleStateSimulator{T,S}}
end


"""
    make_simulator(est::Estimator)

Convenience function for simulator creation. Automatically chooses the
appropriate simulator type for the input estimator.
"""
function make_simulator(est::Estimator)
    return SingleStateSimulator(est.sys, est.obs, est.estimate)
end
function make_simulator(mtf::MultiTargetFilter)
    simulator_bank = [make_simulator(filter) for filter in mtf.filter_bank]
    return MultiStateSimulator(simulator_bank)
end


"""
    simulate(sys::AbstractSystem, obs::AbstractObserver, state::AbstractState, t::Real)

Simulate a combined state prediction and observation with noise. Returns two
absolute states. The first represents the "truth" state and the second is a
simulated measurement of that state.
"""
function simulate(sys::AbstractSystem, obs::AbstractObserver,
                  state::AbstractState, t::Real)
   simulated_state = simulate(sys, state, t)
   simulated_measurement = simulate(obs, simulated_state)
   return simulated_state, simulated_measurement
end

"""
    simulate(ssm:SingleStateSimulator, t::Real)

Simulate a state and measurement for a given time using the internal model
and truth state. Advances the simulator's internal truth state to the given
time. Returns a tuple of (true_state, measurement).
"""
function simulate(ssm::SingleStateSimulator, t::Real)
    true_state, measurement = simulate(ssm.sys, ssm.obs, ssm.true_state, t)
    ssm.true_state .= true_state
    return true_state, measurement
end

"""
    simulate(msm:MultiStateSimulator, t::Real)

Simulate a state and measurement for a given time using a bank of internal
models and truth states. Returns a vector of (true_state, measurement) tuples.
"""
function simulate(msm::MultiStateSimulator, t::Real)
    sim_tuples = [simulate(ssm.sys, ssm.obs, ssm.true_state, t) for
                  ssm in msm.simulator_bank]
    return sim_tuples
end
