#
#
#
abstract type MultiTargetFilter{T,S} <: SequentialEstimator{T,S} end


"""
Multi-target filter with Nearest Neighbor data association.
"""
immutable NearestNeighborMTF{T<:AbstractFloat, S<:AbstractState{T}} <:
                                                          MultiTargetFilter{T,S}
    filter_bank::Vector{SequentialEstimator{T,S}}

    function NearestNeighborMTF(filter::SequentialEstimator{T,S}) where {T,S}
        new{T,S}(Vector{SequentialEstimator{T}}([filter]))
    end
end
function NearestNeighborMTF(filter_bank::Vector)
    if isempty(filter_bank)
        throw(ArgumentError("MTF requires at least one sequential filter."))
    end
    new_mtf = NearestNeighborMTF(filter_bank[1])
    for idx = 2:length(filter_bank)
        add!(new_mtf, filter_bank[idx])
    end
    return new_mtf
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
    for idx = 1:length(mtf.filter_bank)
        if filter === mtf.filter_bank[idx]
            throw(ArgumentError("Attempted to add duplicate filter."))
        end
    end
    push!(mtf.filter_bank, filter)
    return nothing
end
function add!(mtf::MultiTargetFilter, filter::SequentialEstimator)
    throw(ArgumentError("Cannot mix continuous and discrete-time filters."))
end
"""
    add!(mtf::MultiTargetFilter, filter_bank::Array)
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
