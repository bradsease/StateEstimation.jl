#
#
#
abstract type MultiTargetFilter{T} <: SequentialEstimator{T} end


"""

"""
immutable NearestNeighborMTF{T<:AbstractFloat} <: MultiTargetFilter{T}
    filter_bank::Vector{SequentialEstimator{T}}

    function NearestNeighborMTF(filter_bank::Vector{SequentialEstimator{T}}
        ) where T
        new{T}(filter_bank)
    end
    function NearestNeighborMTF(filter::SequentialEstimator{T}) where T
        new{T}(Vector{SequentialEstimator{T}}([filter_bank]))
    end
end
function NearestNeighborMTF()
    NearestNeighborMTF(Vector{SequentialEstimator{Float64}}([]))
end


"""
    add!(mtf::MultiTargetFilter{T}, filter::SequentialEstimator{T})
"""
function add!{T}(mtf::MultiTargetFilter{T}, filter::SequentialEstimator{T})
    push!(mtf.filter_bank, filter)
end
"""
    add!(mtf::MultiTargetFilter{T}, filter_bank::Vector{SequentialEstimator{T}})
"""
function add!{T}(mtf::MultiTargetFilter{T}, filter_bank::Array)
    for idx = 1:length(filter_bank)
        add!(mtf, filter_bank[idx])
    end
end
