abstract type AbstractField{T, N, NF, Vals} <: AbstractArray{T, N} end
abstract type ElementField{T, N, NN, NE, Vals} <: AbstractField{T, N, NN, Vals} end
abstract type NodalField{T, N, NF, NN, Vals} <: AbstractField{T, N, NF, Vals} end
abstract type QuadratureField{T, N, NF, NQ, NE, Vals} <: AbstractField{T, N, NF, Vals} end

Base.eltype(::Type{AbstractField{T, N, NF, Vals}}) where {T, N, NF, Vals} = T
num_fields(::AbstractField{T, N, NF, Vals}) where {T, N, NF, Vals} = NF

num_elements(::ElementField{T, N, NN, NE, Vals}) where {T, N, NN, NE, Vals} = NE
num_elements(::QuadratureField{T, N, NF, NQ, NE, Vals}) where {T, N, NF, NQ, NE, Vals} = NE
num_nodes(::NodalField{T, N, NF, NN, Vals}) where {T, N, NF, NN, Vals} = NN
num_nodes_per_element(field::ElementField) = num_fields(field)
num_q_points(::QuadratureField{T, N, NF, NQ, NE, Vals}) where {T, N, NF, NQ, NE, Vals} = NQ

###############################################

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

function SimpleNodalField{NF, NN}(vals::Matrix{<:Number}) where {NF, NN}
  @assert size(vals) == (NF, NN)
  SimpleNodalField{eltype(vals), 2, NF, NN, typeof(vals)}(vals)
end

function SimpleNodalField{NF, NN, T}(::UndefInitializer) where {NF, NN, T}
  vals = Matrix{T}(undef, NF, NN)
  return SimpleNodalField{T, 2, NF, NN, typeof(vals)}(vals)
end

###############################################################################

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

function VectorizedNodalField{NF, NN}(vals::Vector{<:Number}) where {NF, NN}
  @assert length(vals) == NF * NN
  new_vals = vec(vals)
  VectorizedNodalField{eltype(new_vals), 2, NF, NN, typeof(new_vals)}(new_vals)
end

function VectorizedNodalField{NF, NN}(vals::Matrix{<:Number}) where {NF, NN}
  @assert size(vals) == (NF, NN)
  new_vals = vec(vals)
  VectorizedNodalField{eltype(new_vals), 2, NF, NN, typeof(new_vals)}(new_vals)
end

function VectorizedNodalField{NF, NN, T}(::UndefInitializer) where {NF, NN, T}
  vals = Vector{T}(undef, NF * NN)
  return VectorizedNodalField{T, 2, NF, NN, typeof(vals)}(vals)
end

NodalField{NF, NN, Vector}(vals::Matrix{<:Number}) where {NF, NN}    = VectorizedNodalField{NF, NN}(vals)
NodalField{NF, NN, Matrix}(vals::Matrix{<:Number}) where {NF, NN}    = SimpleNodalField{NF, NN}(vals)
NodalField{NF, NN, Vector}(vals::Vector{<:Number}) where {NF, NN}    = VectorizedNodalField{NF, NN}(vals)
NodalField{NF, NN, Vector, T}(::UndefInitializer)  where {NF, NN, T} = VectorizedNodalField{NF, NN, T}(undef)
NodalField{NF, NN, Matrix, T}(::UndefInitializer)  where {NF, NN, T} = SimpleNodalField{NF, NN, T}(undef)

###############################################################################

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

function SimpleElementField{NN, NE}(vals::Matrix{<:Number}) where {NN, NE}
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

##########################################################################################

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

function VectorizedElementField{NN, NE, Vector, T}(::UndefInitializer) where {NN, NE, T}
  vals = Vector{T}(undef, NN * NE)
  return VectorizedElementField{T, 2, NN, NE, typeof(vals)}(vals)
end

# function VectorizedElementField{NN, NE, StructVector, T}(::UndefInitializer) where {NN, NE, T}
#   vals = StructVector{T}(undef, NN * NE)
#   return VectorizedElementField{T, 2, NN, NE, typeof(vals)}(vals)
# end

##########################################################################################


ElementField{NN, NE, Matrix}(vals::Matrix{<:Number})      where {NN, NE}    = SimpleElementField{NN, NE}(vals)
ElementField{NN, NE, Vector}(vals::Matrix{<:Number})      where {NN, NE}    = VectorizedElementField{NN, NE}(vals)
ElementField{NN, NE, Matrix, T}(::UndefInitializer)       where {NN, NE, T} = SimpleElementField{NN, NE, Matrix, T}(undef)
ElementField{NN, NE, StructArray, T}(::UndefInitializer)  where {NN, NE, T} = SimpleElementField{NN, NE, StructArray, T}(undef)
ElementField{NN, NE, StructVector, T}(::UndefInitializer) where {NN, NE, T} = SimpleElementField{NN, NE, StructVector, T}(undef)
ElementField{NN, NE, Vector, T}(::UndefInitializer)       where {NN, NE, T} = VectorizedElementField{NN, NE, Vector, T}(undef)

##########################################################################################

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

function SimpleQuadratureField{1, NQ, NE}(vals::Matrix{<:Number}) where {NQ, NE}
  @assert size(vals) == (NQ, NE)
  SimpleQuadratureField{eltype(vals), 2, 1, NQ, NE, typeof(vals)}(vals)
end

function SimpleQuadratureField{1, NQ, NE, Matrix, T}(::UndefInitializer) where {NQ, NE, T <: Number}
  vals = Matrix{T}(undef, NQ, NE)
  return SimpleQuadratureField{T, 2, 1, NQ, NE, typeof(vals)}(vals)
end

function SimpleQuadratureField{1, NQ, NE, Matrix, T}(::UndefInitializer) where {NQ, NE, T <: AbstractArray}
  vals = Matrix{T}(undef, NQ, NE)
  return SimpleQuadratureField{T, 2, length(T), NQ, NE, typeof(vals)}(vals)
end

function SimpleQuadratureField{NF, NQ, NE, StructArray, T}(::UndefInitializer) where {NF, NQ, NE, T}
  @assert length(T) == NF
  vals = StructArray{T}(undef, NQ, NE)
  return SimpleQuadratureField{T, 2, length(T), NQ, NE, typeof(vals)}(vals)
end

##########################################################################################

struct VectorizedQuadratureField{
  T, N, NF, NQ, NE, Vals <: AbstractArray{T, 1}
} <: QuadratureField{T, N, NF, NQ, NE, Vals}
  vals::Vals
end

Base.IndexStyle(::Type{<:VectorizedQuadratureField}) = IndexLinear()
Base.axes(field::VectorizedQuadratureField) = (Base.OneTo(num_q_points(field)), Base.OneTo(num_elements(field)))
Base.getindex(field::VectorizedQuadratureField, e::Int) = getindex(field.vals, e)
function Base.getindex(field::VectorizedQuadratureField, q::Int, e::Int) 
  @assert q > 0 && q <= num_q_points(field)
  @assert e > 0 && e <= num_elements(field)
  getindex(field.vals, (e - 1) * num_q_points(field) + q)
end
Base.setindex!(field::VectorizedQuadratureField, v, e::Int) = setindex!(field.vals, v, e)
function Base.setindex!(field::VectorizedQuadratureField, v, q::Int, e::Int)
  @assert q > 0 && q <= num_q_points(field)
  @assert e > 0 && e <= num_elements(field)
  setindex!(field.vals, v, (e - 1) * num_q_points(field) + q)
end
Base.size(::VectorizedQuadratureField{T, N, NF, NQ, NE, V}) where {T, N, NF, NQ, NE, V} = (NQ, NE)

function VectorizedQuadratureField{1, NQ, NE}(vals::Matrix{<:Number}) where {NQ, NE}
  @assert size(vals) == (NQ, NE)
  new_vals = vec(vals)
  VectorizedQuadratureField{eltype(new_vals), 2, 1, NQ, NE, typeof(new_vals)}(new_vals)
end

function VectorizedQuadratureField{1, NQ, NE, Vector, T}(::UndefInitializer) where {NQ, NE, T <: Number}
  vals = Vector{T}(undef, NQ * NE)
  return VectorizedQuadratureField{T, 2, 1, NQ, NE, typeof(vals)}(vals)
end

function VectorizedQuadratureField{1, NQ, NE, Vector, T}(::UndefInitializer) where {NQ, NE, T <: AbstractArray}
  vals = Vector{T}(undef, NQ * NE)
  return VectorizedQuadratureField{T, 2, length(T), NQ, NE, typeof(vals)}(vals)
end

function VectorizedQuadratureField{NF, NQ, NE, StructVector, T}(::UndefInitializer) where {NF, NQ, NE, T}
  @assert length(T) == NF
  vals = StructVector{T}(undef, NQ * NE)
  return VectorizedQuadratureField{T, 2, length(T), NQ, NE, typeof(vals)}(vals)
end

###############################################################################

QuadratureField{NF, NQ, NE, Matrix}(vals::Matrix{<:Number})      where {NF, NQ, NE}    = SimpleQuadratureField{NF, NQ, NE}(vals)
QuadratureField{NF, NQ, NE, Vector}(vals::Matrix{<:Number})      where {NF, NQ, NE}    = VectorizedQuadratureField{NF, NQ, NE}(vals)
QuadratureField{NF, NQ, NE, Matrix, T}(::UndefInitializer)       where {NF, NQ, NE, T} = SimpleQuadratureField{NF, NQ, NE, Matrix, T}(undef)
QuadratureField{NF, NQ, NE, StructArray, T}(::UndefInitializer)  where {NF, NQ, NE, T} = SimpleQuadratureField{NF, NQ, NE, StructArray, T}(undef)
QuadratureField{NF, NQ, NE, StructVector, T}(::UndefInitializer) where {NF, NQ, NE, T} = VectorizedQuadratureField{NF, NQ, NE, StructVector, T}(undef)
QuadratureField{NF, NQ, NE, Vector, T}(::UndefInitializer)       where {NF, NQ, NE, T} = VectorizedQuadratureField{NF, NQ, NE, Vector, T}(undef)
