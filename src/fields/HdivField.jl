"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
Implementation of fields that live in Hdiv spaces.
"""
struct HdivField{T, D, NF} <: AbstractContinuousField{T, D, NF}
    data::D
end

"""
$(TYPEDSIGNATURES)
"""
function HdivField(data::M) where M <: AbstractMatrix
    NF = size(data, 1)
    data = vec(data)
    return HdivField{eltype(data), typeof(data), NF}(data)
end

function Base.zeros(::Type{<:HdivField}, nf::Int, ne::Int)
    data = zeros(nf * ne)
    return HdivField{eltype(data), typeof(data), nf}(data)
end
