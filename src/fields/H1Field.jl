"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
Implementation of fields that live on nodes.
"""
struct H1Field{T, D, NF} <: AbstractContinuousField{T, D, NF}
  data::D
end

"""
$(TYPEDSIGNATURES)
"""
function H1Field(data::M) where M <: AbstractMatrix
  NF = size(data, 1)
  data = vec(data)
  return H1Field{eltype(data), typeof(data), NF}(data)
end

function Base.zeros(::Type{<:H1Field}, nf::Int, ne::Int)
  data = zeros(nf * ne)
  return H1Field{eltype(data), typeof(data), nf}(data)
end
