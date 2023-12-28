struct VectorizedNodalField{
  T, N, NF, NN, Vals <: AbstractArray{T, 1}
} <: NodalField{T, N, NF, NN, Vals}
  vals::Vals
end

Base.IndexStyle(::Type{<:VectorizedNodalField}) = IndexLinear()
Base.axes(::VectorizedNodalField{T, 1, NF, NN, V}) where {T, NF, NN, V <: DenseArray}  = (Base.OneTo(NN),)
Base.axes(::VectorizedNodalField{T, 2, NF, NN, V}) where {T, NF, NN, V <: DenseArray}  = (Base.OneTo(NF), Base.OneTo(NN))
Base.axes(::VectorizedNodalField{T, N, NF, NN, V}) where {T, N, NF, NN, V <: StructArray} = (Base.OneTo(NN),)
Base.getindex(field::VectorizedNodalField, n::Int) = getindex(field.vals, n)
function Base.getindex(field::VectorizedNodalField{T, 2, NF, NN, V}, d::Int, n::Int) where {T, NF, NN, V <: DenseArray}
  @assert d > 0 && d <= num_fields(field)
  @assert n > 0 && n <= num_nodes(field)
  getindex(field.vals, (n - 1) * num_fields(field) + d)
end
Base.setindex!(field::VectorizedNodalField, v, n::Int) = setindex!(field.vals, v, n)
function Base.setindex!(field::VectorizedNodalField{T, 2, NF, NN, V}, v, d::Int, n::Int) where {T, NF, NN, V <: DenseArray}
  @assert d > 0 && d <= num_fields(field)
  @assert n > 0 && n <= num_nodes(field)
  setindex!(field.vals, v, (n - 1) * num_fields(field) + d)
end
Base.size(::VectorizedNodalField{T, 1, NF, NN, V}) where {T, NF, NN, V <: DenseArray} = (NN,)
Base.size(::VectorizedNodalField{T, 2, NF, NN, V}) where {T, NF, NN, V <: DenseArray} = (NF, NN)
Base.size(::VectorizedNodalField{T, N, NF, NN, V}) where {T, N, NF, NN, V <: StructArray}   = (NN,)

function VectorizedNodalField{NF, NN}(vals::V) where {NF, NN, V <: AbstractArray{<:Number, 1}}
  @assert length(vals) == NF * NN
  VectorizedNodalField{eltype(vals), 2, NF, NN, typeof(vals)}(vals)
end

function VectorizedNodalField{NF, NN}(vals::M) where {NF, NN, M <: AbstractArray{<:Number, 2}}
  @assert size(vals) == (NF, NN)
  new_vals = vec(vals)
  VectorizedNodalField{eltype(new_vals), 2, NF, NN, typeof(new_vals)}(new_vals)
end

function VectorizedNodalField{NF, NN}(vals::V) where {NF, NN, V <: AbstractArray{<:AbstractArray, 1}}
  @assert length(eltype(V)) == NF
  @assert size(vals) == (NN,)
  VectorizedNodalField{eltype(vals), 1, NF, NN, typeof(vals)}(vals)
end

function VectorizedNodalField{NF, NN, T}(::UndefInitializer) where {NF, NN, T <: Number}
  vals = Vector{T}(undef, NF * NN)
  return VectorizedNodalField{T, 2, NF, NN, typeof(vals)}(vals)
end

function VectorizedNodalField{NF, NN, T}(::UndefInitializer) where {NF, NN, T <: AbstractArray}
  @assert length(T) == NF
  vals = Vector{T}(undef, NN)
  return VectorizedNodalField{T, 1, NF, NN, typeof(vals)}(vals)
end

function VectorizedNodalField{NF, NN, StructArray, T}(::UndefInitializer) where {NF, NN, T}
  @assert length(T) == NF
  vals = StructArray{T}(undef, NN)
  return VectorizedNodalField{T, 1, length(T), NN, typeof(vals)}(vals)
end
