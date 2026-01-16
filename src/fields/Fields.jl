"""
$(TYPEDEF)
Thin wrapper that subtypes ```AbstractArray``` and serves
as the base ```Field``` type
"""
abstract type AbstractField{T, N, D <: AbstractArray{T, 1}} <: AbstractArray{T, N} end
"""
$(TYPEDSIGNATURES)
"""
Base.unique(field::AbstractField) = unique(field.data)
"""
$(TYPEDSIGNATURES)
"""
Base.fill!(field::AbstractField{T, N, D}, v::T) where {T, N, D} = fill!(field.data, v)
"""
$(TYPEDSIGNATURES)
"""
KA.get_backend(field::AbstractField) = KA.get_backend(field.data)

abstract type AbstractContinuousField{T, D <: AbstractArray{T, 1}, NF} <: AbstractField{T, 2, D} end

function Adapt.adapt_structure(to, field::AbstractContinuousField{T, D, NF}) where {T, D, NF}
  data = adapt(to, field.data)
  type = typeof(field).name.name
  return eval(type){T, typeof(data), NF}(data)
end

# minimal abstractarray interface methods below

function Base.axes(field::AbstractContinuousField{T, D, NF}) where {T, D, NF}
  NN = length(field) ÷ NF
  return (Base.OneTo(NF), Base.OneTo(NN))
end

function Base.getindex(field::AbstractContinuousField, n::Int)
  return getindex(field.data, n)
end

function Base.getindex(field::AbstractContinuousField, d::Int, n::Int)
  @assert d > 0 && d <= num_fields(field)
  @assert n > 0 && n <= num_entities(field)
  getindex(field.data, (n - 1) * num_fields(field) + d)
end

function Base.IndexStyle(::Type{<:AbstractContinuousField}) 
  return IndexLinear()
end

function Base.setindex!(field::AbstractContinuousField{T, D, NF}, v::T, n::Int) where {T, D, NF}
  setindex!(field.data, v, n)
  return nothing
end 

function Base.setindex!(field::AbstractContinuousField{T, D, NF}, v, d::Int, n::Int) where {T, D, NF}
  @assert d > 0 && d <= num_fields(field)
  @assert n > 0 && n <= num_entities(field)
  setindex!(field.data, v, (n - 1) * num_fields(field) + d)
end

function Base.similar(field::AbstractContinuousField)
  data = similar(field.data)
  return typeof(field)(data)
end

function Base.size(field::AbstractContinuousField{T, D, NF}) where {T, D, NF} 
  NN = length(field.data) ÷ NF
  return (NF, NN)
end

# function Base.zeros(type::Type{<:AbstractContinuousField}, nf::Int, ne::Int)
#   @show "Hur"
#   data = zeros(nf * ne)
#   return eval(type.name.name){eltype(data), typeof(data), nf}(data)
# end

"""
$(TYPEDSIGNATURES)
"""
function num_entities(field::AbstractContinuousField{T, D, NF}) where {T, D, NF}
  return length(field.data) ÷ NF
end
"""
$(TYPEDSIGNATURES)
"""
function num_fields(::AbstractContinuousField{T, D, NF}) where {T, D, NF}
  return NF
end

abstract type AbstractDiscontinuousField{T, D <: AbstractArray{T, 1}} <: AbstractField{T, 1, D} end

# actual implementations
include("Connectivity.jl")
include("H1Field.jl")
include("HcurlField.jl")
include("HdivField.jl")
include("L2Field.jl")
