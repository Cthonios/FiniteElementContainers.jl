"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
Implementation of fields that live on nodes.
"""
struct NodalField{T, NF, Vals <: AbstractArray{T, 1}, SymIDMap} <: AbstractField{T, 2, NF, Vals, SymIDMap}
  vals::Vals
end

# constructors

"""
$(TYPEDSIGNATURES)
```NodalField{NF, NN}(vals::V) where {NF, NN, V <: AbstractArray{<:Number, 1}}```
"""
function NodalField{NF, NN}(vals::V, syms) where {NF, NN, V <: AbstractArray{<:Number, 1}}
  @assert length(vals) == NF * NN
  @assert length(syms) == NF
  nt = NamedTuple{syms}(1:length(syms))
  NodalField{eltype(vals), NF, typeof(vals), nt}(vals)
end

"""
$(TYPEDSIGNATURES)
```NodalField{NF, NN}(vals::M) where {NF, NN, M <: AbstractArray{<:Number, 2}}```
"""
function NodalField{NF, NN}(vals::M, syms) where {NF, NN, M <: AbstractArray{<:Number, 2}}
  @assert size(vals) == (NF, NN)
  @assert length(syms) == NF
  new_vals = vec(vals)
  nt = NamedTuple{syms}(1:length(syms))
  NodalField{eltype(new_vals), NF, typeof(new_vals), nt}(new_vals)
end

function NodalField{Tup, T}(::UndefInitializer, syms) where {Tup, T}
  NF, NN = Tup
  @assert length(syms) == NF
  nt = NamedTuple{syms}(1:length(syms))
  vals = Vector{T}(undef, NF * NN)
  return NodalField{T, NF, typeof(vals), nt}(vals)
end

function NodalField(vals::M, syms) where M <: AbstractMatrix
  NF = size(vals, 1)
  @assert length(syms) == NF
  vals = vec(vals)
  nt = NamedTuple{syms}(1:length(syms))
  return NodalField{eltype(vals), NF, typeof(vals), nt}(vals)
end

"""
$(TYPEDSIGNATURES)
"""
NodalField{Tup}(vals, syms) where Tup = NodalField{Tup[1], Tup[2]}(vals, syms)

# general base methods
"""
$(TYPEDSIGNATURES)
"""
function Base.similar(field::NodalField{T, NF, Vals, SymIDMap}) where {T, NF, Vals, SymIDMap}
  vals = similar(field.vals)
  return NodalField{T, NF, Vals, SymIDMap}(vals)
end

function Base.zero(::Type{NodalField{T, NF, Vals, SymIDMap}}, n_nodes) where {T, NF, Vals, SymIDMap}
  vals = zeros(T, NF * n_nodes)
  return NodalField{T, NF, typeof(vals), SymIDMap}(vals)
end

# abstract array interface
Base.IndexStyle(::Type{<:NodalField}) = IndexLinear()

function Base.axes(field::NodalField{T, NF, V, SymIDMap}) where {T, NF, V <: DenseArray, SymIDMap}
  NN = length(field) ÷ NF
  return (Base.OneTo(NF), Base.OneTo(NN))
end

Base.getindex(field::NodalField, n::Int) = getindex(field.vals, n)

function Base.getindex(field::NodalField, d::Int, n::Int)
  @assert d > 0 && d <= num_fields(field)
  @assert n > 0 && n <= num_nodes(field)
  getindex(field.vals, (n - 1) * num_fields(field) + d)
end

function Base.getindex(field::NodalField, sym::Symbol, n::Int)
  d = _sym_id_map(field, sym)
  return getindex(field, d, n)
end

function Base.getindex(field::NodalField, sym::Symbol)
  d = _sym_id_map(field, sym)
  return field[d, :]
end

function Base.getindex(field::NodalField, sym::Symbol, ::Colon)
  d = _sym_id_map(field, sym)
  return field[d, :]
end


Base.setindex!(field::NodalField, v, n::Int) = setindex!(field.vals, v, n)

function Base.setindex!(field::NodalField{T, NF, V, SymIDMap}, v, d::Int, n::Int) where {T, NF, V <: DenseArray, SymIDMap}
  @assert d > 0 && d <= num_fields(field)
  @assert n > 0 && n <= num_nodes(field)
  setindex!(field.vals, v, (n - 1) * num_fields(field) + d)
end

# TODO
# function Base.setindex!(field::NodalField, v, sym::Symbol, n::Int)
#   # d = getproperty(_sym_id_map(field), sym)
#   # d = _sym_id_map(field)
# end

function Base.size(field::NodalField{T, NF, V, SymIDMap}) where {T, NF, V <: DenseArray, SymIDMap} 
  NN = length(field.vals) ÷ NF
  return (NF, NN)
end

# TODO
function Base.view(field::NodalField, sym::Symbol)
  d = _sym_id_map(field, sym)
  return view(field, d, :)
end

function Base.view(field::NodalField, sym::Symbol, ::Colon)
  d = _sym_id_map(field, sym)
  return view(field, d, :)
end

# additional methods
"""
$(TYPEDSIGNATURES)
"""
function num_nodes(field::NodalField{T, NF, Vals, SymIDMap}) where {T, NF, Vals, SymIDMap} 
  NN = length(field) ÷ NF
  return NN
end
