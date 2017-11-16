#
#
#
abstract type MultiTargetFilter{T,S} <: SequentialEstimator{T,S} end


"""

"""
immutable NearestNeighborMTF{T<:AbstractFloat, S<:AbstractState{T}} <:
                                                          MultiTargetFilter{T,S}
    filter_bank::Vector{SequentialEstimator{T}}

    function NearestNeighborMTF(filter_bank::Vector{SequentialEstimator{T,S}}
        ) where {T,S}
        new{T,S}(filter_bank)
    end
    function NearestNeighborMTF(filter::SequentialEstimator{T,S}) where {T,S}
        new{T,S}(Vector{SequentialEstimator{T}}([filter]))
    end
end


const DiscreteMTF{T} =
    MultiTargetFilter{T,UncertainDiscreteState{T}} where T
const ContinuousMTF{T} =
    MultiTargetFilter{T,UncertainContinuousState{T}} where T

const DiscreteSequentialEstimator{T} =
    SequentialEstimator{T,UncertainDiscreteState{T}} where T
const ContinuousSequentialEstimator{T} =
    SequentialEstimator{T,UncertainContinuousState{T}} where T


"""
    add!(mtf::MultiTargetFilter{T}, filter::SequentialEstimator{T})
"""
function add!{T,S}(mtf::MultiTargetFilter{T,S},filter::SequentialEstimator{T,S})
    push!(mtf.filter_bank, filter)
    return nothing
end
"""
    add!(mtf::MultiTargetFilter{T}, filter_bank::Vector{SequentialEstimator{T}})
"""
function add!(mtf::MultiTargetFilter, filter_bank::Array)
    for idx = 1:length(filter_bank)
        add!(mtf, filter_bank[idx])
    end
    return nothing
end


"""
    distance(est::SequentialEstimator{T}, z::AbstractAbsoluteState{T})
"""
function distance{T}(est::DiscreteSequentialEstimator{T}, z::DiscreteState{T})
    return distance(predict(est.obs, predict(est.sys, est.estimate, z.t)), z)
end
function distance{T}(est::ContinuousSequentialEstimator{T},
                     z::ContinuousState{T})
    return distance(predict(est.obs, predict(est.sys, est.estimate, z.t)), z)
end


"""
    process!(mtf::MultiTargetFilter{T}, z::AbstractAbsoluteState{T})
"""
function process!{T<:AbstractFloat}(mtf::MultiTargetFilter{T},
                                    z::AbstractAbsoluteState{T})
    distances::Vector{T} = zeros(length(mtf.filter_bank))
    for idx = 1:length(mtf.filter_bank)
        @inbounds distances[idx] .= distance(mtf.filter_bank[idx], z)
    end
    process!(mtf.filter_bank[indmin(distances)], z)
    return nothing
end
