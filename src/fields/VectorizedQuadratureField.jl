struct VectorizedQuadratureField{
  T, N, NF, NQ, NE, Vals <: AbstractArray{T, 1}
} <: QuadratureField{T, N, NF, NQ, NE, Vals}
  vals::Vals
end

Base.IndexStyle(::Type{<:VectorizedQuadratureField}) = IndexLinear()
Base.axes(field::VectorizedQuadratureField) = (Base.OneTo(num_q_points(field)), Base.OneTo(num_elements(field)))
Base.getindex(field::VectorizedQuadratureField, e::Int) = getindex(field.vals, e)
function Base.getindex(field::VectorizedQuadratureField, q::Int, e::Int) 
  @assert q > 0 && q <= num_q_points(field)
  @assert e > 0 && e <= num_elements(field)
  getindex(field.vals, (e - 1) * num_q_points(field) + q)
end
Base.setindex!(field::VectorizedQuadratureField, v, e::Int) = setindex!(field.vals, v, e)
function Base.setindex!(field::VectorizedQuadratureField, v, q::Int, e::Int)
  @assert q > 0 && q <= num_q_points(field)
  @assert e > 0 && e <= num_elements(field)
  setindex!(field.vals, v, (e - 1) * num_q_points(field) + q)
end
Base.size(::VectorizedQuadratureField{T, N, NF, NQ, NE, V}) where {T, N, NF, NQ, NE, V} = (NQ, NE)

function VectorizedQuadratureField{1, NQ, NE}(vals::M) where {NQ, NE, M <: AbstractArray{<:Number, 2}}
  @assert size(vals) == (NQ, NE)
  new_vals = vec(vals)
  VectorizedQuadratureField{eltype(new_vals), 2, 1, NQ, NE, typeof(new_vals)}(new_vals)
end

function VectorizedQuadratureField{1, NQ, NE}(vals::V) where {NQ, NE, V <: AbstractArray{<:Number, 1}}
  @assert length(vals) == NQ * NE
  VectorizedQuadratureField{eltype(vals), 2, 1, NQ, NE, typeof(vals)}(vals)
end

function VectorizedQuadratureField{NF, NQ, NE}(vals::S) where {NF, NQ, NE, S <: StructVector}
  @assert size(vals) == (NQ * NE,)
  VectorizedQuadratureField{eltype(vals), 2, NF, NQ, NE, typeof(vals)}(vals)
end

function VectorizedQuadratureField{1, NQ, NE, Vector, T}(::UndefInitializer) where {NQ, NE, T <: Number}
  vals = Vector{T}(undef, NQ * NE)
  return VectorizedQuadratureField{T, 2, 1, NQ, NE, typeof(vals)}(vals)
end

function VectorizedQuadratureField{1, NQ, NE, Vector, T}(::UndefInitializer) where {NQ, NE, T <: AbstractArray}
  vals = Vector{T}(undef, NQ * NE)
  return VectorizedQuadratureField{T, 2, length(T), NQ, NE, typeof(vals)}(vals)
end

function VectorizedQuadratureField{NF, NQ, NE, StructVector, T}(::UndefInitializer) where {NF, NQ, NE, T}
  @assert length(T) == NF
  vals = StructVector{T}(undef, NQ * NE)
  return VectorizedQuadratureField{T, 2, length(T), NQ, NE, typeof(vals)}(vals)
end
