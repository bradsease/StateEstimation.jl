#
# Data archiving for state estimators.
#
using Plots

immutable EstimatorHistory{T<:AbstractFloat}
    states::Vector{AbstractUncertainState{T}}
    residuals::Vector{AbstractUncertainState{T}}
end

function EstimatorHistory()
    EstimatorHistory{Float64}([], [])
end

function plot_archive{T}(archive::EstimatorHistory{T})
    t = archive.states[1].t
    data::Array{T,2} = archive.states[1].x'
    cov::Array{T,2} = diag(chol(Hermitian(archive.states[1].P)))'

    for i = 1:length(archive.states)
        t = vcat(t, archive.states[i].t)
        data = vcat(data, archive.states[i].x')
        cov = vcat(cov, diag(chol(Hermitian(archive.states[i].P)))')
    end

    scatter(t, data, markersize=1, markercolor="black",
            layout=(length(archive.states[1].x), 1), legend=false)
    plot!(t, data+3*cov, linestyle=:dot, color="gray")
    plot!(t, data-3*cov, linestyle=:dot, color="gray")
end
