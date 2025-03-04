"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
Implementation of fields that live on elements.
"""
struct L2ElementField{T, NN, Vals <: AbstractArray{T, 1}, SymIDMap} <: AbstractField{T, 2, NN, Vals, SymIDMap}
  vals::Vals
end

# constructors
"""
$(TYPEDSIGNATURES)
"""
function L2ElementField{NF, NN}(vals::V, syms) where {NF, NN, V <: AbstractArray{<:Number, 1}}
  @assert length(vals) == NF * NN
  @assert length(syms) == NF
  nt = NamedTuple{syms}(1:length(syms))
  L2ElementField{eltype(vals), NF, typeof(vals), nt}(vals)
end

"""
$(TYPEDSIGNATURES)
"""
function L2ElementField{NN, NE}(vals::M, syms) where {NN, NE, M <: AbstractArray{<:Number, 2}}
  @assert size(vals) == (NN, NE)
  @assert length(syms) == NN
  new_vals = vec(vals)
  nt = NamedTuple{syms}(1:length(syms))
  L2ElementField{eltype(new_vals), NN, typeof(new_vals), nt}(new_vals)
end

function L2ElementField{Tup, T}(::UndefInitializer, syms) where {Tup, T}
  NN, NE = Tup
  vals = Vector{T}(undef, NN * NE)
  nt = NamedTuple{syms}(1:length(syms))
  return L2ElementField{T, NN, typeof(vals), nt}(vals)
end

"""
$(TYPEDSIGNATURES)
"""
L2ElementField{Tup}(vals, syms) where Tup = L2ElementField{Tup[1], Tup[2]}(vals, syms)

function L2ElementField(vals::M, syms) where M <: AbstractMatrix
  NN = size(vals, 1)
  vals = vec(vals)
  nt = NamedTuple{syms}(1:length(syms))
  return L2ElementField{eltype(vals), NN, typeof(vals), nt}(vals)
end

# general base methods
"""
$(TYPEDSIGNATURES)
"""
function Base.similar(field::L2ElementField{T, NN, Vals, SymIDMap}) where {T, NN, Vals, SymIDMap}
  vals = similar(field.vals)
  return L2ElementField{T, NN, Vals, SymIDMap}(vals)
end

function Base.zero(::Type{L2ElementField{T, NN, Vals, SymIDMap}}, n_elements) where {T, NN, Vals, SymIDMap}
  vals = zeros(T, NN * n_elements)
  return L2ElementField{T, NN, typeof(vals), SymIDMap}(vals)
end

# abstract array interface
Base.IndexStyle(::Type{<:L2ElementField}) = IndexLinear()

function Base.axes(field::L2ElementField{T, NN, V, SymIDMap}) where {T, NN, V <: DenseArray, SymIDMap}
  NE = length(field) ÷ NN
  return (Base.OneTo(NN), Base.OneTo(NE))
end

Base.getindex(field::L2ElementField, n::Int) = getindex(field.vals, n)

function Base.getindex(field::L2ElementField, d::Int, n::Int)
  @assert d > 0 && d <= num_fields(field)
  @assert n > 0 && n <= num_elements(field)
  getindex(field.vals, (n - 1) * num_fields(field) + d)
end

function Base.getindex(field::L2ElementField, sym::Symbol, n::Int)
  d = _sym_id_map(field, sym)
  return getindex(field, d, n)
end

function Base.getindex(field::L2ElementField, sym::Symbol)
  d = _sym_id_map(field, sym)
  return field[d, :]
end

function Base.getindex(field::L2ElementField, sym::Symbol, ::Colon)
  d = _sym_id_map(field, sym)
  return field[d, :]
end

Base.setindex!(field::L2ElementField, v, n::Int) = setindex!(field.vals, v, n)

function Base.setindex!(field::L2ElementField{T, NN, V, SymIDMap}, v, d::Int, n::Int) where {T, NN, V <: DenseArray, SymIDMap}
  @assert d > 0 && d <= num_fields(field)
  @assert n > 0 && n <= num_elements(field)
  setindex!(field.vals, v, (n - 1) * num_fields(field) + d)
end

function Base.size(field::L2ElementField{T, NN, V, SymIDMap}) where {T, NN, V <: DenseArray, SymIDMap} 
  NE = length(field.vals) ÷ NN
  return (NN, NE)
end

function Base.view(field::L2ElementField, sym::Symbol)
  d = _sym_id_map(field, sym)
  return view(field, d, :)
end

function Base.view(field::L2ElementField, sym::Symbol, ::Colon)
  d = _sym_id_map(field, sym)
  return view(field, d, :)
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
