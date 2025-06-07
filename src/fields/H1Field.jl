"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
Implementation of fields that live on nodes.
"""
struct H1Field{T, NF, Vals <: AbstractArray{T, 1}, SymIDMap} <: AbstractField{T, 2, NF, Vals, SymIDMap}
  vals::Vals
end

"""
$(TYPEDSIGNATURES)
"""
function H1Field(vals::M, syms) where M <: AbstractMatrix
  NF = size(vals, 1)
  @assert length(syms) == NF
  vals = vec(vals)
  nt = NamedTuple{syms}(1:length(syms))
  return H1Field{eltype(vals), NF, typeof(vals), nt}(vals)
end

# abstract array interface

function Base.getindex(field::H1Field, d::Int, n::Int)
  @assert d > 0 && d <= num_fields(field)
  @assert n > 0 && n <= num_nodes(field)
  getindex(field.vals, (n - 1) * num_fields(field) + d)
end

function Base.setindex!(field::H1Field{T, NF, V, SymIDMap}, v, d::Int, n::Int) where {T, NF, V <: DenseArray, SymIDMap}
  @assert d > 0 && d <= num_fields(field)
  @assert n > 0 && n <= num_nodes(field)
  setindex!(field.vals, v, (n - 1) * num_fields(field) + d)
end

# TODO
# function Base.setindex!(field::H1Field, v, sym::Symbol, n::Int)
#   # d = getproperty(_sym_id_map(field), sym)
#   # d = _sym_id_map(field)
# end

# additional methods
"""
$(TYPEDSIGNATURES)
"""
function num_nodes(field::H1Field{T, NF, Vals, SymIDMap}) where {T, NF, Vals, SymIDMap} 
  NN = length(field) ÷ NF
  return NN
end
