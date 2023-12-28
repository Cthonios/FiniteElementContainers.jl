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
# implementations
include("fields/SimpleElementField.jl")
include("fields/SimpleNodalField.jl")
include("fields/SimpleQuadratureField.jl")
include("fields/VectorizedElementField.jl")
include("fields/VectorizedNodalField.jl")
include("fields/VectorizedQuadratureField.jl")
#
###############################################################################

# Interfaces

# NodalField{NF, NN, Vector}(vals::Matrix{<:Number}) where {NF, NN}    = VectorizedNodalField{NF, NN}(vals)
# NodalField{NF, NN, Matrix}(vals::Matrix{<:Number}) where {NF, NN}    = SimpleNodalField{NF, NN}(vals)
# NodalField{NF, NN, Vector}(vals::Vector{<:Number}) where {NF, NN}    = VectorizedNodalField{NF, NN}(vals)
NodalField{NF, NN, Vector}(vals::M) where {NF, NN, M <: AbstractArray{<:Number, 2}}    = VectorizedNodalField{NF, NN}(vals)
NodalField{NF, NN, Matrix}(vals::M) where {NF, NN, M <: AbstractArray{<:Number, 2}}    = SimpleNodalField{NF, NN}(vals)
NodalField{NF, NN, Vector}(vals::V) where {NF, NN, V <: AbstractArray{<:Number, 1}}    = VectorizedNodalField{NF, NN}(vals)
NodalField{NF, NN, Vector, T}(::UndefInitializer)  where {NF, NN, T} = VectorizedNodalField{NF, NN, T}(undef)
NodalField{NF, NN, Matrix, T}(::UndefInitializer)  where {NF, NN, T} = SimpleNodalField{NF, NN, T}(undef)
NodalField{NF, NN, StructArray, T}(::UndefInitializer) where {NF, NN, T} = VectorizedNodalField{NF, NN, StructArray, T}(undef)

##########################################################################################

ElementField{NN, NE, Matrix}(vals::Matrix{<:Number})      where {NN, NE}    = SimpleElementField{NN, NE}(vals)
ElementField{NN, NE, Vector}(vals::Matrix{<:Number})      where {NN, NE}    = VectorizedElementField{NN, NE}(vals)
ElementField{NN, NE, Matrix, T}(::UndefInitializer)       where {NN, NE, T} = SimpleElementField{NN, NE, Matrix, T}(undef)
ElementField{NN, NE, StructArray, T}(::UndefInitializer)  where {NN, NE, T} = SimpleElementField{NN, NE, StructArray, T}(undef)
ElementField{NN, NE, StructVector, T}(::UndefInitializer) where {NN, NE, T} = SimpleElementField{NN, NE, StructVector, T}(undef)
ElementField{NN, NE, Vector, T}(::UndefInitializer)       where {NN, NE, T} = VectorizedElementField{NN, NE, Vector, T}(undef)

###############################################################################

QuadratureField{NF, NQ, NE, Matrix}(vals::Matrix{<:Number})      where {NF, NQ, NE}    = SimpleQuadratureField{NF, NQ, NE}(vals)
QuadratureField{NF, NQ, NE, Vector}(vals::Matrix{<:Number})      where {NF, NQ, NE}    = VectorizedQuadratureField{NF, NQ, NE}(vals)
QuadratureField{NF, NQ, NE, Matrix, T}(::UndefInitializer)       where {NF, NQ, NE, T} = SimpleQuadratureField{NF, NQ, NE, Matrix, T}(undef)
QuadratureField{NF, NQ, NE, StructArray, T}(::UndefInitializer)  where {NF, NQ, NE, T} = SimpleQuadratureField{NF, NQ, NE, StructArray, T}(undef)
QuadratureField{NF, NQ, NE, StructVector, T}(::UndefInitializer) where {NF, NQ, NE, T} = VectorizedQuadratureField{NF, NQ, NE, StructVector, T}(undef)
QuadratureField{NF, NQ, NE, Vector, T}(::UndefInitializer)       where {NF, NQ, NE, T} = VectorizedQuadratureField{NF, NQ, NE, Vector, T}(undef)
