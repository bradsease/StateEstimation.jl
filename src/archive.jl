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
    plot_archive(archive::EstimatorHistory)

Plot the contents of an estimator history archive.
"""
function plot_archive(archive::EstimatorHistory)
    t = archive.states[1].t
    data = archive.states[1].x'
    cov = sqrt.(diag(archive.states[1].P))'

    for i = 1:length(archive.states)
        t = vcat(t, archive.states[i].t)
        data = vcat(data, archive.states[i].x')
        cov = vcat(cov, sqrt.(diag(archive.states[i].P))')
    end

    scatter(t, data, markersize=1, markercolor="black",
            layout=(length(archive.states[1].x), 1), legend=false)
    plot!(t, data+3*cov, linestyle=:dot, color="gray")
    plot!(t, data-3*cov, linestyle=:dot, color="gray")
end
