struct SimpleElementField{
  T, N, NN, NE, Vals #<: AbstractArray{T, 2}
} <: ElementField{T, N, NN, NE, Vals}
  vals::Vals
end

Base.IndexStyle(::Type{<:SimpleElementField}) = IndexLinear()

Base.axes(field::SimpleElementField) = axes(field.vals)
Base.getindex(field::SimpleElementField, n::Int) = getindex(field.vals, n)
Base.setindex!(field::SimpleElementField, v, n::Int) = setindex!(field.vals, v, n)
Base.size(::SimpleElementField{T, N, NN, NE, V}) where {T, N, NN, NE, V} = (NN, NE)

function SimpleElementField{NN, NE}(vals::M) where {NN, NE, M <: AbstractArray{<:Number, 2}}
  @assert size(vals) == (NN, NE)
  SimpleElementField{eltype(vals), 2, NN, NE, typeof(vals)}(vals)
end

function SimpleElementField{NN, NE, Matrix, T}(::UndefInitializer) where {NN, NE, T}
  vals = Matrix{T}(undef, NN, NE)
  return SimpleElementField{T, 2, NN, NE, typeof(vals)}(vals)
end

function SimpleElementField{NN, NE, StructVector, T}(::UndefInitializer) where {NN, NE, T}
  # @assert length(T) == NN
  vals = StructVector{T}(undef, NE)
  return SimpleElementField{T, 1, NN, NE, typeof(vals)}(vals)
end

function SimpleElementField{NN, NE, StructArray, T}(::UndefInitializer) where {NN, NE, T}
  vals = StructArray{T}(undef, NN, NE)
  return SimpleElementField{T, 2, NN, NE, typeof(vals)}(vals)
end

function Base.similar(field::SimpleElementField{T, N, NN, NE, Vals}) where {T, N, NN, NE, Vals}
  vals = similar(field.vals)
  return SimpleElementField{T, N, NN, NE, Vals}(vals)
end

function Base.zero(::Type{SimpleElementField{T, N, NN, NE, Vals}}) where {T, N, NN, NE, Vals <: AbstractMatrix}
  vals = zeros(T, NN, NE)
  return SimpleElementField{T, N, NN, NE, Vals}(vals)
end

function Base.zero(::Type{SimpleElementField{T, N, NN, NE, Vals}}) where {T, N, NN, NE, Vals <: StructVector}
  vals = StructVector{T}(undef, NE)
  for e in axes(vals, 1)
    vals[e] = zero(T)
  end
  return SimpleElementField{T, N, NN, NE, Vals}(vals)
end

function Base.zero(::Type{SimpleElementField{T, N, NN, NE, Vals}}) where {T, N, NN, NE, Vals <: StructArray}
  vals = StructVector{T}(undef, NN, NE)
  for e in axes(vals, 2)
    for q in axes(vals, 1)
      vals[q, e] = zero(T)
    end
  end
  return SimpleElementField{T, N, NN, NE, Vals}(vals)
end

function Base.zero(field::SimpleElementField{T, N, NN, NE, Vals}) where {T, N, NN, NE, Vals <: AbstractArray}
  vals = similar(field.vals)
  return SimpleElementField{T, N, NN, NE, Vals}(vals)
end

function Base.zero(::SimpleElementField{T, N, NN, NE, Vals}) where {T, N, NN, NE, Vals <: StructVector}
  vals = StructVector{T}(undef, NE)
  for e in axes(vals, 1)
    vals[e] = zero(T)
  end
  return SimpleElementField{T, N, NN, NE, Vals}(vals)
end