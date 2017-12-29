#
# Convenience simulators for estimator types
#


"""
Single-state simulator type
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
Multi-state simulator type
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
    simulate(ssm:SingleStateSimulator, t)

Simulate a state and measurement for a given time using the internal model
and truth state. Advances the internal truth state to the given time.
"""
function simulate(ssm::SingleStateSimulator, t)
    true_state, measurement = simulate(ssm.sys, ssm.obs, ssm.true_state, t)
    ssm.true_state .= true_state
    return true_state, measurement
end

"""
    simulate(msm:MultiStateSimulator, t)

Simulate a state and measurement for a given time using a bank of internal
models and truth states. Returns a vector of (true_state, measurement) tuples.
"""
function simulate(msm::MultiStateSimulator, t)
    sim_tuples = [simulate(ssm.sys, ssm.obs, ssm.true_state, t) for
                  ssm in msm.simulator_bank]
    return sim_tuples
end
