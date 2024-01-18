struct SimpleNodalField{
  T, N, NF, NN, Vals <: AbstractArray{T, 2}
} <: NodalField{T, N, NF, NN, Vals}
  vals::Vals
end

Base.IndexStyle(::Type{<:SimpleNodalField}) = IndexLinear()

Base.axes(field::SimpleNodalField) = axes(field.vals)
Base.getindex(field::SimpleNodalField, n::Int) = getindex(field.vals, n)
Base.setindex!(field::SimpleNodalField, v, n::Int) = setindex!(field.vals, v, n)
Base.size(::SimpleNodalField{T, N, NF, NN, V}) where {T, N, NF, NN, V} = (NF, NN)

function SimpleNodalField{NF, NN}(vals::V) where {NF, NN, V <: AbstractArray{<:Number, 2}}
  @assert size(vals) == (NF, NN) "NF = $NF, NN = $NN"
  SimpleNodalField{eltype(vals), 2, NF, NN, typeof(vals)}(vals)
end

function SimpleNodalField{NF, NN, T}(::UndefInitializer) where {NF, NN, T}
  vals = Matrix{T}(undef, NF, NN)
  return SimpleNodalField{T, 2, NF, NN, typeof(vals)}(vals)
end

function Base.similar(field::SimpleNodalField{T, N, NF, NN, Vals}) where {T, N, NF, NN, Vals}
  vals = similar(field.vals)
  return SimpleNodalField{T, N, NF, NN, Vals}(vals)
end

function Base.zero(::Type{SimpleNodalField{T, N, NF, NN, Vals}}) where {T, N, NF, NN, Vals}
  vals = zeros(T, NF, NN)
  return SimpleNodalField{T, N, NF, NN, Vals}(vals)
end

function Base.zero(::SimpleNodalField{T, N, NF, NN, Vals}) where {T, N, NF, NN, Vals}
  vals = zeros(T, NF, NN)
  return SimpleNodalField{T, N, NF, NN, Vals}(vals)
end