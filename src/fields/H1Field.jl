"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
Implementation of fields that live on nodes.
"""
struct H1Field{T, D, NF} <: AbstractField{T, 2, D, NF}
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

# abstract array interface

function Base.getindex(field::H1Field, d::Int, n::Int)
  @assert d > 0 && d <= num_fields(field)
  @assert n > 0 && n <= num_nodes(field)
  getindex(field.data, (n - 1) * num_fields(field) + d)
end

function Base.setindex!(field::H1Field{T, D, NF}, v, d::Int, n::Int) where {T, D, NF}
  @assert d > 0 && d <= num_fields(field)
  @assert n > 0 && n <= num_nodes(field)
  setindex!(field.data, v, (n - 1) * num_fields(field) + d)
end

# additional methods
"""
$(TYPEDSIGNATURES)
"""
function num_nodes(field::H1Field{T, D, NF}) where {T, D, NF} 
  NN = length(field) ÷ NF
  return NN
end
