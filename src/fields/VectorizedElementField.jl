struct VectorizedElementField{
  T, N, NN, NE, Vals <: AbstractArray{T, 1}
} <: ElementField{T, N, NN, NE, Vals}
  vals::Vals
end

Base.IndexStyle(::Type{<:VectorizedElementField}) = IndexLinear()
Base.axes(field::VectorizedElementField) = (Base.OneTo(num_nodes_per_element(field)), Base.OneTo(num_elements(field)))
Base.getindex(field::VectorizedElementField, e::Int) = getindex(field.vals, e)
function Base.getindex(field::VectorizedElementField, n::Int, e::Int) 
  @assert n > 0 && n <= num_nodes_per_element(field)
  @assert e > 0 && e <= num_elements(field)
  getindex(field.vals, (e - 1) * num_nodes_per_element(field) + n)
end
Base.setindex!(field::VectorizedElementField, v, e::Int) = setindex!(field.vals, v, e)
function Base.setindex!(field::VectorizedElementField, v, n::Int, e::Int)
  @assert n > 0 && n <= num_nodes_per_element(field)
  @assert e > 0 && e <= num_elements(field)
  setindex!(field.vals, v, (e - 1) * num_nodes_per_element(field) + n)
end
Base.size(::VectorizedElementField{T, N, NN, NE, V}) where {T, N, NN, NE, V} = (NN, NE)

function VectorizedElementField{NN, NE}(vals::Matrix{<:Number}) where {NN, NE}
  @assert size(vals) == (NN, NE)
  new_vals = vec(vals)
  VectorizedElementField{eltype(new_vals), 2, NN, NE, typeof(new_vals)}(new_vals)
end

function VectorizedElementField{NF, NN}(vals::V) where {NF, NN, V <: AbstractArray{<:Number, 1}}
  @assert length(vals) == NF * NN
  VectorizedElementField{eltype(vals), 2, NF, NN, typeof(vals)}(vals)
end

function VectorizedElementField{NN, NE, Vector, T}(::UndefInitializer) where {NN, NE, T}
  vals = Vector{T}(undef, NN * NE)
  return VectorizedElementField{T, 2, NN, NE, typeof(vals)}(vals)
end

# function VectorizedElementField{NN, NE, StructVector, T}(::UndefInitializer) where {NN, NE, T}
#   vals = StructVector{T}(undef, NN * NE)
#   return VectorizedElementField{T, 2, NN, NE, typeof(vals)}(vals)
# end