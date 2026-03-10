"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
Implementation of fields that live in Hdiv spaces.
"""
struct HcurlField{T, D, NF} <: AbstractContinuousField{T, D, NF}
    data::D
end

"""
$(TYPEDSIGNATURES)
"""
function HcurlField(data::M) where M <: AbstractMatrix
    NF = size(data, 1)
    data = vec(data)
    return HcurlField{eltype(data), typeof(data), NF}(data)
end

function Base.zeros(::Type{<:HcurlField}, nf::Int, ne::Int)
    data = zeros(nf * ne)
    return HcurlField{eltype(data), typeof(data), nf}(data)
end
