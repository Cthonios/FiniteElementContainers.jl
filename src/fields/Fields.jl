"""
$(TYPEDEF)
Thin wrapper that subtypes ```AbstractArray``` and serves
as the base ```Field``` type
"""
abstract type AbstractField{T, N, NF, Vals, SymIDMap} <: AbstractArray{T, N} end

"""
$(TYPEDSIGNATURES)
"""
function KA.get_backend(field::AbstractField)
  return KA.get_backend(field.vals)
end

"""
$(TYPEDSIGNATURES)
"""
Base.fill!(field::AbstractField{T, N, NF, V, S}, v::T) where {T, N, NF, V, S} = 
fill!(field.vals, v)

"""
$(TYPEDSIGNATURES)
"""
Base.names(::AbstractField{T, N, NF, Vals, SymIDMap}) where {T, N, NF, Vals, SymIDMap} = keys(SymIDMap)

"""
$(TYPEDSIGNATURES)
"""
num_fields(::AbstractField{T, N, NF, Vals, SymIDMap}) where {T, N, NF, Vals, SymIDMap} = NF

"""
$(TYPEDSIGNATURES)
"""
_sym_id_map(::AbstractField{T, N, NF, Vals, SymIDMap}, sym::Symbol) where {T, N, NF, Vals, SymIDMap} = getproperty(SymIDMap, sym)

# minimal abstractarray interface methods below

function Base.axes(field::AbstractField{T, 2, NF, V, S}) where {T, NF, V, S}
  NN = length(field) ÷ NF
  return (Base.OneTo(NF), Base.OneTo(NN))
end

function Base.getindex(field::AbstractField, n::Int)
  return getindex(field.vals, n)
end

function Base.getindex(field::AbstractField{T, 2, NF, V, S}, sym::Symbol) where {T, NF, V, S}
  d = _sym_id_map(field, sym)
  return field[d, :]
end

function Base.getindex(field::AbstractField{T, 3, NF, V, S}, sym::Symbol) where {T, NF, V, S}
  d = _sym_id_map(field, sym)
  return field[d, :, :]
end

function Base.getindex(field::AbstractField{T, 2, NF, V, S}, sym::Symbol, ::Colon) where {T, NF, V, S}
  d = _sym_id_map(field, sym)
  return field[d, :]
end

function Base.getindex(field::AbstractField{T, 3, NF, V, S}, sym::Symbol, ::Colon, ::Colon) where {T, NF, V, S}
  d = _sym_id_map(field, sym)
  return field[d, :, :]
end

function Base.getindex(field::AbstractField{T, 2, NF, V, S}, sym::Symbol, n::Int) where {T, NF, V, S}
  d = _sym_id_map(field, sym)
  return field[d, n]
end

function Base.getindex(field::AbstractField{T, 3, NF, V, S}, sym::Symbol, m::Int, n::Int) where {T, NF, V, S}
  d = _sym_id_map(field, sym)
  return field[d, m, n]
end

function Base.IndexStyle(::Type{<:AbstractField}) 
  return IndexLinear()
end

function Base.setindex!(field::AbstractField{T, N, NF, V, S}, v::T, n::Int) where {T, N, NF, V, S}
  setindex!(field.vals, v, n)
  return nothing
end 

function Base.similar(field::AbstractField)
  vals = similar(field.vals)
  return typeof(field)(vals)
end

function Base.size(field::AbstractField{T, 2, NF, V, SymIDMap}) where {T, NF, V <: DenseArray, SymIDMap} 
  NN = length(field.vals) ÷ NF
  return (NF, NN)
end

function Base.view(field::AbstractField{T, 2, NF, V, S}, sym::Symbol) where {T, NF, V, S}
  d = _sym_id_map(field, sym)
  return view(field, d, :)
end

function Base.view(field::AbstractField{T, 2, NF, V, S}, sym::Symbol, ::Colon) where {T, NF, V, S}
  d = _sym_id_map(field, sym)
  return view(field, d, :)
end

function Base.view(field::AbstractField{T, 3, NF, V, S}, sym::Symbol) where {T, NF, V, S}
  d = _sym_id_map(field, sym)
  return view(field, d, :, :)
end

function Base.view(field::AbstractField{T, 3, NF, V, S}, sym::Symbol, ::Colon, ::Colon) where {T, NF, V, S}
  d = _sym_id_map(field, sym)
  return view(field, d, :, :)
end

# actual implementations
# include("ElementField.jl")
include("H1Field.jl")
include("L2ElementField.jl")
include("L2QuadratureField.jl")

# some specialization
include("Connectivity.jl")
