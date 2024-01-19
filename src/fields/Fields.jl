"""
"""
abstract type AbstractField{T, N, NF, Vals} <: AbstractArray{T, N} end

"""
"""
abstract type ElementField{T, N, NN, NE, Vals} <: AbstractField{T, N, NN, Vals} end

"""
"""
abstract type NodalField{T, N, NF, NN, Vals} <: AbstractField{T, N, NF, Vals} end

"""
"""
abstract type QuadratureField{T, N, NF, NQ, NE, Vals} <: AbstractField{T, N, NF, Vals} end

"""
"""
Base.eltype(::Type{AbstractField{T, N, NF, Vals}}) where {T, N, NF, Vals} = T

"""
"""
num_fields(::AbstractField{T, N, NF, Vals}) where {T, N, NF, Vals} = NF

"""
"""
num_elements(::ElementField{T, N, NN, NE, Vals}) where {T, N, NN, NE, Vals} = NE

"""
"""
num_elements(::QuadratureField{T, N, NF, NQ, NE, Vals}) where {T, N, NF, NQ, NE, Vals} = NE

"""
"""
num_nodes(::NodalField{T, N, NF, NN, Vals}) where {T, N, NF, NN, Vals} = NN

"""
"""
num_nodes_per_element(field::ElementField) = num_fields(field)

"""
"""
num_q_points(::QuadratureField{T, N, NF, NQ, NE, Vals}) where {T, N, NF, NQ, NE, Vals} = NQ

###############################################
# implementations
include("SimpleElementField.jl")
include("SimpleNodalField.jl")
include("SimpleQuadratureField.jl")
include("VectorizedElementField.jl")
include("VectorizedNodalField.jl")
include("VectorizedQuadratureField.jl")
#
###############################################################################

# Interfaces

NodalField{NF, NN, Vector}(vals::M) where {NF, NN, M <: AbstractArray{<:Number, 2}} = VectorizedNodalField{NF, NN}(vals)
NodalField{NF, NN, Matrix}(vals::M) where {NF, NN, M <: AbstractArray{<:Number, 2}} = SimpleNodalField{NF, NN}(vals)
NodalField{NF, NN, Vector}(vals::V) where {NF, NN, V <: AbstractArray{<:Number, 1}} = VectorizedNodalField{NF, NN}(vals)
NodalField{NF, NN, Vector, T}(::UndefInitializer)  where {NF, NN, T} = VectorizedNodalField{NF, NN, T}(undef)
NodalField{NF, NN, Matrix, T}(::UndefInitializer)  where {NF, NN, T <: Number} = SimpleNodalField{NF, NN, T}(undef)
NodalField{NF, NN, StructArray, T}(::UndefInitializer) where {NF, NN, T} = VectorizedNodalField{NF, NN, StructArray, T}(undef)
NodalField{Tup, A, T}(::UndefInitializer) where {Tup, A, T} = NodalField{Tup[1], Tup[2], A, T}(undef)
NodalField{Tup, A}(vals::M) where {Tup, A, M <: AbstractArray} = NodalField{Tup[1], Tup[2], A}(vals)
 
# Failed attempts at type stable stuff below don't think this will work
# NodalField{V1, V2, A}(vals::M) where {V1}

# NodalField(vals::M, ::Val{NF}, ::Val{NN}) where {M, NF, NN} = 
# # SimpleNodalField{Float64, 2, NF, NN, Matrix{Float64}}(vals)
# NodalField{NF, NN, Matrix}(vals)



# # function NodalField{A}(vals::M) where {A, M <: AbstractMatrix}
# function make_nodal_field(vals::M) where M <: AbstractMatrix
#   # return NodalField{Val(size(vals, 1)), Val(size(vals, 2)), A}(vals)
#   NF = Val{(size(vals, 1))}()#::Int
#   NN = Val{(size(vals, 2))}()#::Int
#   # NF = Val(2)
#   # NN = Val(10)
#   return NodalField(vals, NF, NN)
# end

# function NodalField{A}(vals::M) where {M <: AbstractMatrix}

# end

##########################################################################################

ElementField{NN, NE, Vector}(vals::M) where {NN, NE, M <: AbstractArray{<:Number, 2}}    = VectorizedElementField{NN, NE}(vals)
ElementField{NN, NE, Matrix}(vals::M) where {NN, NE, M <: AbstractArray{<:Number, 2}}    = SimpleElementField{NN, NE}(vals)
ElementField{NN, NE, Vector}(vals::V) where {NN, NE, V <: AbstractArray{<:Number, 1}}    = VectorizedElementField{NN, NE}(vals)
ElementField{NN, NE, Vector, T}(::UndefInitializer)  where {NN, NE, T} = VectorizedElementField{NN, NE, T}(undef)
ElementField{NN, NE, Matrix, T}(::UndefInitializer)  where {NN, NE, T <: Number} = SimpleElementField{NN, NE, Matrix, T}(undef)
ElementField{NN, NE, StructArray, T}(::UndefInitializer) where {NN, NE, T} = VectorizedElementField{NN, NE, StructArray, T}(undef)
ElementField{Tup, A, T}(::UndefInitializer) where {Tup, A, T} = ElementField{Tup[1], Tup[2], A, T}(undef)
ElementField{Tup, A}(vals::M) where {Tup, A, M <: AbstractArray} = ElementField{Tup[1], Tup[2], A}(vals)

###############################################################################

QuadratureField{NF, NQ, NE, Matrix}(vals::Matrix{<:Number})      where {NF, NQ, NE}    = SimpleQuadratureField{NF, NQ, NE}(vals)
QuadratureField{NF, NQ, NE, Vector}(vals::Matrix{<:Number})      where {NF, NQ, NE}    = VectorizedQuadratureField{NF, NQ, NE}(vals)
QuadratureField{NF, NQ, NE, Matrix, T}(::UndefInitializer)       where {NF, NQ, NE, T} = SimpleQuadratureField{NF, NQ, NE, Matrix, T}(undef)
QuadratureField{NF, NQ, NE, StructArray, T}(::UndefInitializer)  where {NF, NQ, NE, T} = SimpleQuadratureField{NF, NQ, NE, StructArray, T}(undef)
QuadratureField{NF, NQ, NE, StructVector, T}(::UndefInitializer) where {NF, NQ, NE, T} = VectorizedQuadratureField{NF, NQ, NE, StructVector, T}(undef)
QuadratureField{NF, NQ, NE, Vector, T}(::UndefInitializer)       where {NF, NQ, NE, T} = VectorizedQuadratureField{NF, NQ, NE, Vector, T}(undef)
QuadratureField{Tup, A, T}(::UndefInitializer) where {Tup, A, T} = QuadratureField{Tup[1], Tup[2], Tup[3], A, T}(undef)
QuadratureField{Tup, A}(vals::M) where {Tup, A, M <: AbstractArray} = QuadratureField{Tup[1], Tup[2], Tup[3], A}(vals) 
