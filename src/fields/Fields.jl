"""
$(TYPEDEF)
Thin wrapper that subtypes ```AbstractArray``` and serves
as the base ```Field``` type
"""
abstract type AbstractField{T, N, D <: AbstractArray{T, 1}, NF} <: AbstractArray{T, N} end

"""
$(TYPEDSIGNATURES)
"""
Base.fill!(field::AbstractField{T, N, D, NF}, v::T) where {T, N, D, NF} = fill!(field.data, v)

# minimal abstractarray interface methods below

function Base.axes(field::AbstractField{T, 2, D, NF}) where {T, D, NF}
  NN = length(field) ÷ NF
  return (Base.OneTo(NF), Base.OneTo(NN))
end

function Base.getindex(field::AbstractField, n::Int)
  return getindex(field.data, n)
end

function Base.IndexStyle(::Type{<:AbstractField}) 
  return IndexLinear()
end

function Base.setindex!(field::AbstractField{T, N, D, NF}, v::T, n::Int) where {T, N, D, NF}
  setindex!(field.data, v, n)
  return nothing
end 

function Base.similar(field::AbstractField)
  data = similar(field.data)
  return typeof(field)(data)
end

function Base.size(field::AbstractField{T, 2, D, NF}) where {T, D, NF} 
  NN = length(field.data) ÷ NF
  return (NF, NN)
end

function Base.unique(field::AbstractField)
  return unique(field.data)
end

"""
$(TYPEDSIGNATURES)
"""
function KA.get_backend(field::AbstractField)
  return KA.get_backend(field.data)
end

"""
$(TYPEDSIGNATURES)
"""
function num_fields(::AbstractField{T, N, D, NF}) where {T, N, D, NF}
  return NF
end

# actual implementations
include("H1Field.jl")
include("L2ElementField.jl")
include("L2QuadratureField.jl")

include("L2Field.jl")

# some specialization
include("Connectivity.jl")
