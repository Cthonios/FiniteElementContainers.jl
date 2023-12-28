struct VectorizedNodalField{
  T, N, NF, NN, Vals <: AbstractArray{T, 1}
} <: NodalField{T, N, NF, NN, Vals}
  vals::Vals
end

Base.IndexStyle(::Type{<:VectorizedNodalField}) = IndexLinear()
Base.axes(field::VectorizedNodalField) = (Base.OneTo(num_fields(field)), Base.OneTo(num_nodes(field)))
Base.getindex(field::VectorizedNodalField, n::Int) = getindex(field.vals, n)
function Base.getindex(field::VectorizedNodalField, d::Int, n::Int) 
  @assert d > 0 && d <= num_fields(field)
  @assert n > 0 && n <= num_nodes(field)
  getindex(field.vals, (n - 1) * num_fields(field) + d)
end
Base.setindex!(field::VectorizedNodalField, v, n::Int) = setindex!(field.vals, v, n)
function Base.setindex!(field::VectorizedNodalField, v, d::Int, n::Int)
  @assert d > 0 && d <= num_fields(field)
  @assert n > 0 && n <= num_nodes(field)
  setindex!(field.vals, v, (n - 1) * num_fields(field) + d)
end
Base.size(::VectorizedNodalField{T, N, NF, NN, V}) where {T, N, NF, NN, V} = (NF, NN)

function VectorizedNodalField{NF, NN}(vals::V) where {NF, NN, V <: AbstractArray{<:Number, 1}}
  @assert length(vals) == NF * NN
  VectorizedNodalField{eltype(vals), 2, NF, NN, typeof(vals)}(vals)
end

function VectorizedNodalField{NF, NN}(vals::M) where {NF, NN, M <: AbstractArray{<:Number, 2}}
  @assert size(vals) == (NF, NN)
  new_vals = vec(vals)
  VectorizedNodalField{eltype(new_vals), 2, NF, NN, typeof(new_vals)}(new_vals)
end

function VectorizedNodalField{NF, NN, T}(::UndefInitializer) where {NF, NN, T}
  vals = Vector{T}(undef, NF * NN)
  return VectorizedNodalField{T, 2, NF, NN, typeof(vals)}(vals)
end

function VectorizedNodalField{NF, NN, StructArray, T}(::UndefInitializer) where {NF, NN, T}
  @assert length(T) == NF
  vals = StructArray{T}(undef, NN)
  return VectorizedNodalField{T, 1, length(T), NN, typeof(vals)}(vals)
end