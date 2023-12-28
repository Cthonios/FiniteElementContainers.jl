struct VectorizedElementField{
  T, N, NN, NE, Vals <: AbstractArray{T, 1}
} <: ElementField{T, N, NN, NE, Vals}
  vals::Vals
end

Base.IndexStyle(::Type{<:VectorizedElementField}) = IndexLinear()

Base.axes(::VectorizedElementField{T, 1, NN, NE, V}) where {T, NN, NE, V <: DenseArray}  = (Base.OneTo(NE),)
Base.axes(::VectorizedElementField{T, 2, NN, NE, V}) where {T, NN, NE, V <: DenseArray}  = (Base.OneTo(NN), Base.OneTo(NE))
Base.axes(::VectorizedElementField{T, N, NN, NE, V}) where {T, N, NN, NE, V <: StructArray} = (Base.OneTo(NE),)
Base.getindex(field::VectorizedElementField, e::Int) = getindex(field.vals, e)
function Base.getindex(field::VectorizedElementField{T, 2, NN, NE, V}, n::Int, e::Int) where {T, NN, NE, V <: DenseArray}
  @assert n > 0 && n <= num_nodes_per_element(field)
  @assert e > 0 && e <= num_elements(field)
  getindex(field.vals, (e - 1) * num_nodes_per_element(field) + n)
end
Base.setindex!(field::VectorizedElementField, v, e::Int) = setindex!(field.vals, v, e)
function Base.setindex!(field::VectorizedElementField{T, 2, NN, NE, V}, v, n::Int, e::Int) where {T, NN, NE, V <: DenseArray}
  @assert n > 0 && n <= num_nodes_per_element(field)
  @assert e > 0 && e <= num_elements(field)
  setindex!(field.vals, v, (e - 1) * num_nodes_per_element(field) + n)
end
Base.size(::VectorizedElementField{T, 1, NN, NE, V}) where {T, NN, NE, V <: DenseArray} = (NE,)
Base.size(::VectorizedElementField{T, 2, NN, NE, V}) where {T, NN, NE, V <: DenseArray} = (NN, NE)
Base.size(::VectorizedElementField{T, N, NN, NE, V}) where {T, N, NN, NE, V <: StructArray} = (NE,)

function VectorizedElementField{NN, NE}(vals::Matrix{<:Number}) where {NN, NE}
  @assert size(vals) == (NN, NE)
  new_vals = vec(vals)
  VectorizedElementField{eltype(new_vals), 2, NN, NE, typeof(new_vals)}(new_vals)
end

function VectorizedElementField{NN, NE}(vals::V) where {NN, NE, V <: AbstractArray{<:Number, 1}}
  @assert length(vals) == NN * NE
  VectorizedElementField{eltype(vals), 2, NN, NE, typeof(vals)}(vals)
end

function VectorizedElementField{NN, NE}(vals::V) where {NN, NE, V <: AbstractArray{<:AbstractArray, 1}}
  @assert length(eltype(V)) == NN
  @assert size(vals) == (NE,)
  VectorizedElementField{eltype(vals), 1, NN, NE, typeof(vals)}(vals)
end

function VectorizedElementField{NN, NE, T}(::UndefInitializer) where {NN, NE, T <: Number}
  vals = Vector{T}(undef, NN * NE)
  return VectorizedElementField{T, 2, NN, NE, typeof(vals)}(vals)
end

function VectorizedElementField{NN, NE, T}(::UndefInitializer) where {NN, NE, T <: AbstractArray}
  @assert length(T) == NN
  vals = Vector{T}(undef, NE)
  return VectorizedElementField{T, 1, NN, NE, typeof(vals)}(vals)
end

function VectorizedElementField{NN, NE, StructArray, T}(::UndefInitializer) where {NN, NE, T}
  @assert length(T) == NN
  vals = StructArray{T}(undef, NE)
  return VectorizedElementField{T, 1, length(T), NE, typeof(vals)}(vals)
end
