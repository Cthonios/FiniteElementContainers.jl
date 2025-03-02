struct QuadratureField{T, NF, NQ, Vals <: AbstractArray{T, 1}} <: AbstractField{T, 3, NF, Vals}
  vals::Vals
end

function QuadratureField{NF, NQ, NE}(vals::V) where {NF, NQ, NE, V <: AbstractArray{<:Number, 1}}
  @assert length(vals) == NF * NQ * NE
  # TODO
end

function QuadratureField{NF, NQ, NE}(vals::V) where {NF, NQ, NE, V <: AbstractArray{<:Number, 3}}
  @assert size(vals) == (NF, NQ, NE)
  new_vals = vec(vals)
  return QuadratureField{eltype(new_vals), NF, NQ, typeof(new_vals)}(new_vals)
end

