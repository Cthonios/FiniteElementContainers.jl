"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
Implementation of fields that live on elements.
"""
struct L2QuadratureField{T, NF, NQ, Vals <: AbstractArray{T, 1}, SymIDMap} <: AbstractField{T, 3, NF, Vals, SymIDMap}
  vals::Vals
end

# currently has shenanigans when vals has zero fields
# will create a field that is 0 x NQ x 0 when it should
# be 0 x NQ x NF. Likely due to the vec(vals) call
function L2QuadratureField(vals::A, syms) where A <: AbstractArray{<:Number, 3}
  NF, NQ = size(vals, 1), size(vals, 2)
  @assert length(syms) == NF
  nt = NamedTuple{syms}(1:length(syms))
  vals = vec(vals)
  L2QuadratureField{eltype(vals), NF, NQ, typeof(vals), nt}(vals)
end

# TODO add more constructors

# abstract array interface
Base.IndexStyle(::Type{<:L2QuadratureField}) = IndexLinear()

function Base.axes(field::L2QuadratureField{T, NF, NQ, V, SymIDMap}) where {T, NF, NQ, V, SymIDMap}
  if NF == 0
    NE = length(field.vals) ÷ NQ
  else
    NE = length(field.vals) ÷ NF ÷ NQ
  end
  return (Base.OneTo(NF), Base.OneTo(NQ), Base.OneTo(NE))
end

Base.getindex(field::L2QuadratureField, n::Int) = getindex(field.vals, n)

function Base.getindex(field::L2QuadratureField{T, NF, NQ, V, SymIDMap}, n::Int, q::Int, e::Int) where {T, NF, NQ, V, SymIDMap}
  getindex(field.vals, (e - 1) * NQ + (q - 1) * NF + n)
end

function Base.getindex(field::L2QuadratureField, sym::Symbol, q::Int, e::Int)
  d = _sym_id_map(field, sym)
  return getindex(field, d, q, e)
end

function Base.getindex(field::L2QuadratureField, sym::Symbol)
  d = _sym_id_map(field, sym)
  return field[d, :, :]
end

function Base.getindex(field::L2QuadratureField, sym::Symbol, ::Colon, ::Colon)
  d = _sym_id_map(field, sym)
  return field[d, :, :]
end 

Base.setindex!(field::L2QuadratureField, v, n::Int) = setindex!(field.vals, v, n)
function Base.setindex!(field::L2QuadratureField{T, NF, NQ, A, S}, v, n::Int, q::Int, e::Int) where {T, NF, NQ, A, S}
  setindex!(field.vals, v, (e - 1) * NQ + (q - 1) * NF + n)
end

function Base.size(field::L2QuadratureField{T, NF, NQ, A, S}) where {T, NF, NQ, A, S} 
  if NF == 0
    (NF, NQ, length(field.vals) ÷ NQ)
  else
    (NF, NQ, length(field.vals) ÷ NF ÷ NQ)
  end
end

function Base.view(field::L2QuadratureField, sym::Symbol)
  d = _sym_id_map(field, sym)
  return view(field, d, :, :)
end

function Base.view(field::L2QuadratureField, sym::Symbol, ::Colon, ::Colon)
  d = _sym_id_map(field, sym)
  return view(field, d, :, :)
end

function num_elements(field::L2QuadratureField{T, NF, NQ, A, S}) where {T, NF, NQ, A, S}
  NE = length(field) ÷ NF ÷ NQ
  return NE
end
