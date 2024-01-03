struct SimpleQuadratureField{
  T, N, NF, NQ, NE, Vals <: AbstractArray{T, 2}
} <: QuadratureField{T, N, NF, NQ, NE, Vals}
  vals::Vals
end

Base.IndexStyle(::Type{<:SimpleQuadratureField}) = IndexLinear()

Base.axes(field::SimpleQuadratureField) = axes(field.vals)
Base.getindex(field::SimpleQuadratureField, n::Int) = getindex(field.vals, n)
Base.setindex!(field::SimpleQuadratureField, v, n::Int) = setindex!(field.vals, v, n)
Base.size(::SimpleQuadratureField{T, N, NF, NQ, NE, V}) where {T, N, NF, NQ, NE, V} = (NQ, NE)

function SimpleQuadratureField{1, NQ, NE}(vals::M) where {NQ, NE, M <: AbstractArray{<:Number, 2}}
  @assert size(vals) == (NQ, NE)
  SimpleQuadratureField{eltype(vals), 2, 1, NQ, NE, typeof(vals)}(vals)
end

function SimpleQuadratureField{NF, NQ, NE}(vals::S) where {NF, NQ, NE, S <: StructArray}
  @assert size(vals) == (NQ, NE)
  SimpleQuadratureField{eltype(vals), 2, NF, NQ, NE, typeof(vals)}(vals)
end

function SimpleQuadratureField{1, NQ, NE, Matrix, T}(::UndefInitializer) where {NQ, NE, T <: Number}
  vals = Matrix{T}(undef, NQ, NE)
  return SimpleQuadratureField{T, 2, 1, NQ, NE, typeof(vals)}(vals)
end

function SimpleQuadratureField{1, NQ, NE, Matrix, T}(::UndefInitializer) where {NQ, NE, T <: AbstractArray}
  vals = Matrix{T}(undef, NQ, NE)
  return SimpleQuadratureField{T, 2, length(T), NQ, NE, typeof(vals)}(vals)
end

function SimpleQuadratureField{NF, NQ, NE, StructArray, T}(::UndefInitializer) where {NF, NQ, NE, T}
  @assert length(T) == NF
  vals = StructArray{T}(undef, NQ, NE)
  return SimpleQuadratureField{T, 2, length(T), NQ, NE, typeof(vals)}(vals)
end

function Base.similar(field::SimpleQuadratureField{T, N, NF, NQ, NE, Vals}) where {T, N, NF, NQ, NE, Vals}
  vals = similar(field.vals)
  return SimpleQuadratureField{T, N, NF, NQ, NE, Vals}(vals)
end
