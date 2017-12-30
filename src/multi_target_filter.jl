#
#
#
abstract type MultiTargetFilter{T,S} <: SequentialEstimator{T,S} end


"""
    NearestNeighborMTF(filter::SequentialEstimator)
    NearestNeighborMTF(filter_bank::Vector{SequentialEstimator})

Multi-target filter with nearest-neighbor data association. A multi-target
filter contains a bank of `SequentialEstimator` types representing
currently-tracked targets. To process a measurement, the `NearestNeighborMTF`
computes a Euclidean distance between the input measurement and the next
predicted measurement for all internal estimators. The `SequentialEstimator`
with the best prediction calls its own `process!` method on the measurement.

MultiTargetFilter types require at least one `SequentialEstimator`, but accept
any number in a Vector format at construction-time. Additional estimators can
be added to the MultiTargetFilter at any time with the `add!` method.

All internal `SequentialEstimator` must have the same time format. Attempting to
mix Discrete- and Continuous-time estimators will result in an error. Internal
estimators must also be unique.
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
    add!(mtf::MultiTargetFilter, filter::SequentialEstimator)
    add!(mtf::MultiTargetFilter, filter_bank::Array)

Add one or more `SequentialEstimator` to a `MultiTargetFilter`. All
`SequentialEstimator` in a `MultiTargetFilter` must have the same time format.
Attempting to mix Discrete- and Continuous-time estimators will result in an
error. Internal estimators must also be unique.
"""
function add!(mtf::MultiTargetFilter, filter::SequentialEstimator)
    throw(ArgumentError("Cannot mix continuous and discrete-time filters."))
end
function add!{T,S}(mtf::MultiTargetFilter{T,S},filter::SequentialEstimator{T,S})
    for idx = 1:length(mtf.filter_bank)
        if filter === mtf.filter_bank[idx]
            throw(ArgumentError("Attempted to add duplicate filter."))
        end
    end
    push!(mtf.filter_bank, filter)
    return nothing
end
function add!(mtf::MultiTargetFilter, filter_bank::Array)
    for idx = 1:length(filter_bank)
        add!(mtf, filter_bank[idx])
    end
    return nothing
end


"""
    distance(est::SequentialEstimator, z::AbstractAbsoluteState)
"""
function distance(est::DiscreteSequentialEstimator, z::DiscreteState)
    return distance(predict(est.obs, predict(est.sys, est.estimate, z.t)), z)
end
function distance(est::ContinuousSequentialEstimator, z::ContinuousState)
    return distance(predict(est.obs, predict(est.sys, est.estimate, z.t)), z)
end


"""
    process!(mtf::MultiTargetFilter, z::AbstractAbsoluteState)

Process a measurement with an arbitrary `MultiTargetFilter`. Uses the
appropriate data association method for the particular type of
`MultiTargetFilter`.
"""
function process!(::MultiTargetFilter) end
function process!{T<:AbstractFloat}(mtf::NearestNeighborMTF{T},
                                    zk::AbstractAbsoluteState{T})
    distances = zeros(T, length(mtf.filter_bank))
    for idx = 1:length(mtf.filter_bank)
        @inbounds distances[idx] .= distance(mtf.filter_bank[idx], zk)
    end
    process!(mtf.filter_bank[indmin(distances)], zk)
    return nothing
end
