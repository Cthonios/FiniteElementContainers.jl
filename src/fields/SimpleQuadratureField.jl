"""
$(TYPEDEF)
"""
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

"""
```SimpleQuadratureField{1, NQ, NE}(vals::M) where {NQ, NE, M <: AbstractArray{<:Number, 2}}```
"""
function SimpleQuadratureField{1, NQ, NE}(vals::M) where {NQ, NE, M <: AbstractArray{<:Number, 2}}
  @assert size(vals) == (NQ, NE)
  SimpleQuadratureField{eltype(vals), 2, 1, NQ, NE, typeof(vals)}(vals)
end

"""
```SimpleQuadratureField{NF, NQ, NE}(vals::S) where {NF, NQ, NE, S <: StructArray}```
"""
function SimpleQuadratureField{NF, NQ, NE}(vals::S) where {NF, NQ, NE, S <: StructArray}
  @assert size(vals) == (NQ, NE)
  SimpleQuadratureField{eltype(vals), 2, NF, NQ, NE, typeof(vals)}(vals)
end

"""
```SimpleQuadratureField{1, NQ, NE, Matrix, T}(::UndefInitializer) where {NQ, NE, T <: Number}```
"""
function SimpleQuadratureField{1, NQ, NE, Matrix, T}(::UndefInitializer) where {NQ, NE, T <: Number}
  vals = Matrix{T}(undef, NQ, NE)
  return SimpleQuadratureField{T, 2, 1, NQ, NE, typeof(vals)}(vals)
end

"""
```SimpleQuadratureField{1, NQ, NE, Matrix, T}(::UndefInitializer) where {NQ, NE, T <: AbstractArray}```
"""
function SimpleQuadratureField{1, NQ, NE, Matrix, T}(::UndefInitializer) where {NQ, NE, T <: AbstractArray}
  vals = Matrix{T}(undef, NQ, NE)
  return SimpleQuadratureField{T, 2, length(T), NQ, NE, typeof(vals)}(vals)
end

"""
```SimpleQuadratureField{NF, NQ, NE, StructArray, T}(::UndefInitializer) where {NF, NQ, NE, T}```
"""
function SimpleQuadratureField{NF, NQ, NE, StructArray, T}(::UndefInitializer) where {NF, NQ, NE, T}
  @assert length(T) == NF
  vals = StructArray{T}(undef, NQ, NE)
  return SimpleQuadratureField{T, 2, length(T), NQ, NE, typeof(vals)}(vals)
end

"""
$(TYPEDSIGNATURES)
"""
function Base.similar(field::SimpleQuadratureField{T, N, NF, NQ, NE, Vals}) where {T, N, NF, NQ, NE, Vals}
  vals = similar(field.vals)
  return SimpleQuadratureField{T, N, NF, NQ, NE, Vals}(vals)
end

"""
$(TYPEDSIGNATURES)
"""
function Base.zero(::Type{SimpleQuadratureField{T, N, NF, NQ, NE, Vals}}) where {T, N, NF, NQ, NE, Vals <: AbstractMatrix}
  vals = zeros(T, NQ, NE)
  return SimpleQuadratureField{T, N, NF, NQ, NE, Vals}(vals)
end

"""
$(TYPEDSIGNATURES)
"""
function Base.zero(::Type{SimpleQuadratureField{T, N, NF, NQ, NE, Vals}}) where {T, N, NF, NQ, NE, Vals <: StructArray}
  vals = StructArray{T}(undef, NQ, NE)
  for e in axes(vals, 2)
    for q in axes(vals, 1)
      vals[q, e] = zero(T)
    end
  end
  return SimpleQuadratureField{T, N, NF, NQ, NE, Vals}(vals)
end

"""
$(TYPEDSIGNATURES)
"""
function Base.zero(field::SimpleQuadratureField{T, N, NF, NQ, NE, Vals}) where {T, N, NF, NQ, NE, Vals}
  vals = zero(field.vals)
  return SimpleQuadratureField{T, N, NF, NQ, NE, Vals}(vals)
end
