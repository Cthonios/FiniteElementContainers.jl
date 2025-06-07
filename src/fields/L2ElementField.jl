"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
Implementation of fields that live on elements.
"""
struct L2ElementField{T, NN, Vals <: AbstractArray{T, 1}, SymIDMap} <: AbstractField{T, 2, NN, Vals, SymIDMap}
  vals::Vals
end

"""
$(TYPEDSIGNATURES)
"""
function L2ElementField(vals::M, syms) where M <: AbstractMatrix
  NN = size(vals, 1)
  vals = vec(vals)
  nt = NamedTuple{syms}(1:length(syms))
  return L2ElementField{eltype(vals), NN, typeof(vals), nt}(vals)
end

# abstract array interface

function Base.getindex(field::L2ElementField, d::Int, n::Int)
  @assert d > 0 && d <= num_fields(field)
  @assert n > 0 && n <= num_elements(field)
  getindex(field.vals, (n - 1) * num_fields(field) + d)
end

function Base.setindex!(field::L2ElementField{T, NN, V, SymIDMap}, v, d::Int, n::Int) where {T, NN, V <: DenseArray, SymIDMap}
  @assert d > 0 && d <= num_fields(field)
  @assert n > 0 && n <= num_elements(field)
  setindex!(field.vals, v, (n - 1) * num_fields(field) + d)
end

"""
$(TYPEDSIGNATURES)
"""
function num_elements(field::L2ElementField{T, NN, Vals, SymIDMap}) where {T, NN, Vals, SymIDMap}
  NE = length(field) ÷ NN
  return NE
end

"""
$(TYPEDSIGNATURES)
"""
num_nodes_per_element(field::L2ElementField) = num_fields(field)
