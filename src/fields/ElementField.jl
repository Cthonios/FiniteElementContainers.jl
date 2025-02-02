"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
Implementation of fields that live on elements.
"""
struct ElementField{T, NN, Vals <: AbstractArray{T, 1}} <: AbstractField{T, 2, NN, Vals}
  vals::Vals
end

# constructors
"""
$(TYPEDSIGNATURES)
"""
function ElementField{NF, NN}(vals::V) where {NF, NN, V <: AbstractArray{<:Number, 1}}
  @assert length(vals) == NF * NN
  ElementField{eltype(vals), NF, typeof(vals)}(vals)
end

"""
$(TYPEDSIGNATURES)
"""
function ElementField{NN, NE}(vals::M) where {NN, NE, M <: AbstractArray{<:Number, 2}}
  @assert size(vals) == (NN, NE)
  new_vals = vec(vals)
  ElementField{eltype(new_vals), NN, typeof(new_vals)}(new_vals)
end

function ElementField{Tup, T}(::UndefInitializer) where {Tup, T}
  NN, NE = Tup
  vals = Vector{T}(undef, NN * NE)
  return ElementField{T, NN, typeof(vals)}(vals)
end

"""
$(TYPEDSIGNATURES)
"""
ElementField{Tup}(vals) where Tup = ElementField{Tup[1], Tup[2]}(vals)

function ElementField(vals::M) where M <: AbstractMatrix
  NN = size(vals, 1)
  vals = vec(vals)
  return ElementField{eltype(vals), NN, typeof(vals)}(vals)
end

# general base methods
"""
$(TYPEDSIGNATURES)
"""
function Base.similar(field::ElementField{T, NN, Vals}) where {T, NN, Vals}
  vals = similar(field.vals)
  return ElementField{T, NN, Vals}(vals)
end

function Base.zero(::Type{ElementField{T, NN, Vals}}, n_elements) where {T, NN, Vals}
  vals = zeros(T, NN * n_elements)
  return ElementField{T, NN, typeof(vals)}(vals)
end

# abstract array interface
Base.IndexStyle(::Type{<:ElementField}) = IndexLinear()

function Base.axes(field::ElementField{T, NN, V}) where {T, NN, V <: DenseArray}
  NE = length(field) ÷ NN
  return (Base.OneTo(NN), Base.OneTo(NE))
end

Base.getindex(field::ElementField, n::Int) = getindex(field.vals, n)

function Base.getindex(field::ElementField, d::Int, n::Int)
  @assert d > 0 && d <= num_fields(field)
  @assert n > 0 && n <= num_elements(field)
  getindex(field.vals, (n - 1) * num_fields(field) + d)
end

Base.setindex!(field::ElementField, v, n::Int) = setindex!(field.vals, v, n)

function Base.setindex!(field::ElementField{T, NN, V}, v, d::Int, n::Int) where {T, NN, V <: DenseArray}
  @assert d > 0 && d <= num_fields(field)
  @assert n > 0 && n <= num_elements(field)
  setindex!(field.vals, v, (n - 1) * num_fields(field) + d)
end

function Base.size(field::ElementField{T, NN, V}) where {T, NN, V <: DenseArray} 
  NE = length(field.vals) ÷ NN
  return (NN, NE)
end

"""
$(TYPEDSIGNATURES)
"""
function num_elements(field::ElementField{T, NN, Vals}) where {T, NN, Vals}
  NE = length(field) ÷ NN
  return NE
end

"""
$(TYPEDSIGNATURES)
"""
num_nodes_per_element(field::ElementField) = num_fields(field)
