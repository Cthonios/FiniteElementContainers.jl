"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
Implementation of fields that live on elements.
"""
struct L2QuadratureField{T, D, NF, NQ} <: AbstractField{T, 3, D, NF}
  data::D
end

# currently has shenanigans when vals has zero fields
# will create a field that is 0 x NQ x 0 when it should
# be 0 x NQ x NF. Likely due to the vec(vals) call
function L2QuadratureField(data::A) where A <: AbstractArray{<:Number, 3}
  NF, NQ = size(data, 1), size(data, 2)
  data = vec(data)
  return L2QuadratureField{eltype(data), typeof(data), NF, NQ}(data)
end

function Adapt.adapt_structure(to, field::L2QuadratureField{T, D, NF, NQ}) where {T, D, NF, NQ}
  data = adapt(to, field.data)
  return L2QuadratureField{T, typeof(data), NF, NQ}(data)
end

# abstract array interface

function Base.axes(field::L2QuadratureField{T, D, NF, NQ}) where {T, D, NF, NQ}
  if NF == 0
    NE = length(field.data) ÷ NQ
  else
    NE = length(field.data) ÷ NF ÷ NQ
  end
  return (Base.OneTo(NF), Base.OneTo(NQ), Base.OneTo(NE))
end

# TODO these methods aren't consistent with single index
# getindex methods
# function Base.getindex(field::L2QuadratureField{T, NF, NQ, V, SymIDMap}, n::Int, q::Int, e::Int) where {T, NF, NQ, V, SymIDMap}
#   getindex(field.vals, (e - 1) * NQ + (q - 1) * NF + n)
# end

# function Base.setindex!(field::L2QuadratureField{T, NF, NQ, A, S}, v, n::Int, q::Int, e::Int) where {T, NF, NQ, A, S}
#   setindex!(field.vals, v, (e - 1) * NQ + (q - 1) * NF + n)
# end

function Base.setindex!(field::L2QuadratureField{T, D, NF, NQ}, v, n::Int, q::Int, e::Int) where {T, D, NF, NQ}
  setindex!(field.data, v, (e - 1) * NQ + (q - 1) * NF + n)
  return nothing
end

function Base.size(field::L2QuadratureField{T, D, NF, NQ}) where {T, D, NF, NQ} 
  if NF == 0
    (NF, NQ, length(field.data) ÷ NQ)
  else
    (NF, NQ, length(field.data) ÷ NF ÷ NQ)
  end
end

function num_elements(field::L2QuadratureField{T, D, NF, NQ}) where {T, D, NF, NQ}
  NE = length(field) ÷ NF ÷ NQ
  return NE
end
