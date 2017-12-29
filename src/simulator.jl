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

    function SingleStateSimulator(sys::AbstractSystem{T},
                                  obs::AbstractObserver{T},
                                  state::S
                                  ) where {T, S<:AbstractAbsoluteState{T}}
        new{T,S}(sys, obs, state)
    end
end
function SingleStateSimulator(sys::AbstractSystem, obs::AbstractObserver,
                              state::AbstractUncertainState)
    return SingleStateSimulator(sys, obs, sample(state))
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


# """
# Multi-state simulator type
# """
# immutable MultiStateSimulator{T,S<:AbstractUncertainState{T}}
#     true_state::Vector{S}
#
#     function MultiStateSimulator(initial_states::Vector{S}
#                                 ) where {T, S<:AbstractUncertainState{T}}
#         new{T,S}(deepcopy(initial_states))
#     end
# end
# function MultiStateSimulator{T,S<:AbstractUncertainState}(
#     mtf::MultiTargetFilter{T,S})
#
#     initial_states::Vector{S} = []
#     for idx = 1:length(estimator.filter_bank)
#         push!(initial_states, estimator.filter_bank[idx].estimate)
#     end
#
#     return MultiStateSimulator(initial_states)
# end


"""
    make_simulator(est::Estimator)

Convenience function for simulator creation. Automatically chooses the
appropriate simulator type for the input estimator.
"""
function make_simulator(est::SequentialEstimator)
    return SingleStateSimulator(est.sys, est.obs, est.estimate)
end
function make_simulator(est::BatchEstimator)
    return SingleStateSimulator(est.sys, est.obs, est.estimate)
end
