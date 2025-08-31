"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
Implementation of fields that live on elements.
"""
struct L2ElementField{T, D, NF} <: AbstractField{T, 2, D, NF}
  data::D
end

"""
$(TYPEDSIGNATURES)
"""
function L2ElementField(data::M) where M <: AbstractMatrix
  NF = size(data, 1)
  data = vec(data)
  return L2ElementField{eltype(data), typeof(data), NF}(data)
end

# abstract array interface

function Base.getindex(field::L2ElementField, d::Int, n::Int)
  @assert d > 0 && d <= num_fields(field)
  @assert n > 0 && n <= num_elements(field)
  getindex(field.data, (n - 1) * num_fields(field) + d)
end

function Base.setindex!(field::L2ElementField{T, D, NF}, v, d::Int, n::Int) where {T, D, NF}
  @assert d > 0 && d <= num_fields(field)
  @assert n > 0 && n <= num_elements(field)
  setindex!(field.data, v, (n - 1) * num_fields(field) + d)
end

"""
$(TYPEDSIGNATURES)
"""
function num_elements(field::L2ElementField{T, D, NF}) where {T, D, NF}
  NE = length(field) ÷ NF
  return NE
end

"""
$(TYPEDSIGNATURES)
"""
num_nodes_per_element(field::L2ElementField) = num_fields(field)
