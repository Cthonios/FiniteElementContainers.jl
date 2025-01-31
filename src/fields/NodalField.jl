"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
Implementation of fields that live on nodes.
"""
struct NodalField{T, NF, Vals <: AbstractArray{T, 1}} <: AbstractField{T, 2, NF, Vals}
  vals::Vals
end

# constructors

"""
$(TYPEDSIGNATURES)
```NodalField{NF, NN}(vals::V) where {NF, NN, V <: AbstractArray{<:Number, 1}}```
"""
function NodalField{NF, NN}(vals::V) where {NF, NN, V <: AbstractArray{<:Number, 1}}
  @assert length(vals) == NF * NN
  NodalField{eltype(vals), NF, typeof(vals)}(vals)
end

"""
$(TYPEDSIGNATURES)
```NodalField{NF, NN}(vals::M) where {NF, NN, M <: AbstractArray{<:Number, 2}}```
"""
function NodalField{NF, NN}(vals::M) where {NF, NN, M <: AbstractArray{<:Number, 2}}
  @assert size(vals) == (NF, NN)
  new_vals = vec(vals)
  NodalField{eltype(new_vals), NF, typeof(new_vals)}(new_vals)
end

function NodalField{Tup, T}(::UndefInitializer) where {Tup, T}
  NF, NN = Tup
  vals = Vector{T}(undef, NF * NN)
  return NodalField{T, NF, typeof(vals)}(vals)
end

"""
$(TYPEDSIGNATURES)
"""
NodalField{Tup}(vals) where Tup = NodalField{Tup[1], Tup[2]}(vals)

# general base methods
"""
$(TYPEDSIGNATURES)
"""
function Base.similar(field::NodalField{T, NF, Vals}) where {T, NF, Vals}
  vals = similar(field.vals)
  return NodalField{T, NF, Vals}(vals)
end

function Base.zero(::Type{NodalField{T, NF, Vals}}, n_nodes) where {T, NF, Vals}
  vals = zeros(T, NF * n_nodes)
  return NodalField{T, NF, typeof(vals)}(vals)
end

# abstract array interface
Base.IndexStyle(::Type{<:NodalField}) = IndexLinear()

function Base.axes(field::NodalField{T, NF, V}) where {T, NF, V <: DenseArray}
  NN = length(field) ÷ NF
  return (Base.OneTo(NF), Base.OneTo(NN))
end

Base.getindex(field::NodalField, n::Int) = getindex(field.vals, n)

function Base.getindex(field::NodalField, d::Int, n::Int)
  @assert d > 0 && d <= num_fields(field)
  @assert n > 0 && n <= num_nodes(field)
  getindex(field.vals, (n - 1) * num_fields(field) + d)
end

Base.setindex!(field::NodalField, v, n::Int) = setindex!(field.vals, v, n)

function Base.setindex!(field::NodalField{T, NF, V}, v, d::Int, n::Int) where {T, NF, V <: DenseArray}
  @assert d > 0 && d <= num_fields(field)
  @assert n > 0 && n <= num_nodes(field)
  setindex!(field.vals, v, (n - 1) * num_fields(field) + d)
end

function Base.size(field::NodalField{T, NF, V}) where {T, NF, V <: DenseArray} 
  NN = length(field.vals) ÷ NF
  return (NF, NN)
end

# additional methods
"""
$(TYPEDSIGNATURES)
"""
function num_nodes(field::NodalField{T, NF, Vals}) where {T, NF, Vals} 
  NN = length(field) ÷ NF
  return NN
end
