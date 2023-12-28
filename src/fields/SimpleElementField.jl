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
