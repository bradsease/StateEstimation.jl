#
# Data archiving for state estimators.
#
using Plots


"""
    EstimatorHistory(target_estimator::Estimator)

Estimation history archival type.
"""
immutable EstimatorHistory{T<:AbstractFloat, S<:AbstractUncertainState{T}}
    states::Vector{S}
    residuals::Vector{S}

    function EstimatorHistory(target_estimator::Estimator{T,S}) where {T,S}
        new{T,S}([], [])
    end
end


"""
    vectorize_state_history(state_history::Vector)
"""
function vectorize_state_history(state_history::Vector)
    state_length = length(state_history[1].x)
    num_states = length(state_history)

    times = fill(0*state_history[1].t, num_states)
    states = fill(0*state_history[1].x[1], (num_states, state_length))
    sigmas = fill(0*state_history[1].x[1], (num_states, state_length))

    for i = 1:num_states
        times[i] .= state_history[i].t
        states[i,:] .= state_history[i].x
        sigmas[i,:] .= sqrt.(diag(state_history[i].P))
    end

    return times, states, sigmas
end


"""
    plot_state_history(archive::EstimatorHistory)

Plot the state history in an estimator history archive.
"""
function plot_state_history(archive::EstimatorHistory)
    t, data, cov = vectorize_state_history(archive.states)
    scatter(t, data, markersize=1, markercolor="black",
            layout=(length(archive.states[1].x), 1), legend=false)
    plot!(t, data+3*cov, linestyle=:dot, color="gray")
    plot!(t, data-3*cov, linestyle=:dot, color="gray")
end


"""
    plot_residuals(archive::EstimatorHistory)

Plot the residuals in an estimator history archive.
"""
function plot_residuals(archive::EstimatorHistory)
    t, data, cov = vectorize_state_history(archive.residuals)
    scatter(t, data, markersize=1, markercolor="black",
            layout=(length(archive.states[1].x), 1), legend=false)
    plot!(t, data+3*cov, linestyle=:dot, color="gray")
    plot!(t, data-3*cov, linestyle=:dot, color="gray")
end
