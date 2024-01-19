"""
$(TYPEDEF)
Thin wrapper that subtypes ```AbstractArray``` and serves
as the base ```Field``` type
"""
abstract type AbstractField{T, N, NF, Vals} <: AbstractArray{T, N} end

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
Abstract type for implementations of fields that live on elements.

Constructors\n
```ElementField{NN, NE, Vector}(vals::M)                    where {NN, NE, M <: AbstractArray{<:Number, 2}}```\n
```ElementField{NN, NE, Matrix}(vals::M)                    where {NN, NE, M <: AbstractArray{<:Number, 2}}```\n
```ElementField{NN, NE, Vector}(vals::V)                    where {NN, NE, V <: AbstractArray{<:Number, 1}}```\n
```ElementField{NN, NE, Vector, T}(::UndefInitializer)      where {NN, NE, T}```\n
```ElementField{NN, NE, Matrix, T}(::UndefInitializer)      where {NN, NE, T <: Number}```\n
```ElementField{NN, NE, StructArray, T}(::UndefInitializer) where {NN, NE, T}```\n
```ElementField{Tup, A, T}(::UndefInitializer)              where {Tup, A, T}```\n
```ElementField{Tup, A}(vals::M)                            where {Tup, A, M <: AbstractArray}```
"""
abstract type ElementField{T, N, NN, NE, Vals} <: AbstractField{T, N, NN, Vals} end

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
Abstract type for implementations of fields that live on nodes.

Constructors\n
```NodalField{NF, NN, Vector}(vals::M)                    where {NF, NN, M <: AbstractArray{<:Number, 2}}```\n
```NodalField{NF, NN, Matrix}(vals::M)                    where {NF, NN, M <: AbstractArray{<:Number, 2}}```\n
```NodalField{NF, NN, Vector}(vals::V)                    where {NF, NN, V <: AbstractArray{<:Number, 1}}```\n
```NodalField{NF, NN, Vector}(vals::V)                    where {NF, NN, V <: AbstractArray{<:Number, 1}}```\n
```NodalField{NF, NN, Vector, T}(::UndefInitializer)      where {NF, NN, T}```\n
```NodalField{NF, NN, Matrix, T}(::UndefInitializer)      where {NF, NN, T <: Number}```\n
```NodalField{NF, NN, StructArray, T}(::UndefInitializer) where {NF, NN, T}```\n
```NodalField{Tup, A, T}(::UndefInitializer)              where {Tup, A, T}```
"""
abstract type NodalField{T, N, NF, NN, Vals} <: AbstractField{T, N, NF, Vals} end

"""
$(TYPEDEF)
Abstract type for implementations of fields that live on quadrature points.

Constructors\n
```QuadratureField{NF, NQ, NE, Matrix}(vals::Matrix{<:Number})      where {NF, NQ, NE}```\n
```QuadratureField{NF, NQ, NE, Vector}(vals::Matrix{<:Number})      where {NF, NQ, NE}```\n
```QuadratureField{NF, NQ, NE, Matrix, T}(::UndefInitializer)       where {NF, NQ, NE, T}```\n
```QuadratureField{NF, NQ, NE, StructArray, T}(::UndefInitializer)  where {NF, NQ, NE, T}```\n
```QuadratureField{NF, NQ, NE, StructVector, T}(::UndefInitializer) where {NF, NQ, NE, T}```\n
```QuadratureField{NF, NQ, NE, Vector, T}(::UndefInitializer)       where {NF, NQ, NE, T}```\n
```QuadratureField{Tup, A, T}(::UndefInitializer)                   where {Tup, A, T}```\n
```QuadratureField{Tup, A}(vals::M)                                 where {Tup, A, M <: AbstractArray}```
"""
abstract type QuadratureField{T, N, NF, NQ, NE, Vals} <: AbstractField{T, N, NF, Vals} end

"""
$(TYPEDSIGNATURES)
"""
Base.eltype(::Type{AbstractField{T, N, NF, Vals}}) where {T, N, NF, Vals} = T

"""
$(TYPEDSIGNATURES)
"""
num_fields(::AbstractField{T, N, NF, Vals}) where {T, N, NF, Vals} = NF

"""
$(TYPEDSIGNATURES)
"""
num_elements(::ElementField{T, N, NN, NE, Vals}) where {T, N, NN, NE, Vals} = NE

"""
$(TYPEDSIGNATURES)
"""
num_elements(::QuadratureField{T, N, NF, NQ, NE, Vals}) where {T, N, NF, NQ, NE, Vals} = NE

"""
$(TYPEDSIGNATURES)
"""
num_nodes(::NodalField{T, N, NF, NN, Vals}) where {T, N, NF, NN, Vals} = NN

"""
$(TYPEDSIGNATURES)
"""
num_nodes_per_element(field::ElementField) = num_fields(field)

"""
$(TYPEDSIGNATURES)
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

"""
```NodalField{NF, NN, Vector}(vals::M) where {NF, NN, M <: AbstractArray{<:Number, 2}}```
"""
NodalField{NF, NN, Vector}(vals::M) where {NF, NN, M <: AbstractArray{<:Number, 2}} = VectorizedNodalField{NF, NN}(vals)
"""
```NodalField{NF, NN, Matrix}(vals::M) where {NF, NN, M <: AbstractArray{<:Number, 2}}```
"""
NodalField{NF, NN, Matrix}(vals::M) where {NF, NN, M <: AbstractArray{<:Number, 2}} = SimpleNodalField{NF, NN}(vals)
"""
```NodalField{NF, NN, Vector}(vals::V) where {NF, NN, V <: AbstractArray{<:Number, 1}}```
"""
NodalField{NF, NN, Vector}(vals::V) where {NF, NN, V <: AbstractArray{<:Number, 1}} = VectorizedNodalField{NF, NN}(vals)
"""
```NodalField{NF, NN, Vector}(vals::V) where {NF, NN, V <: AbstractArray{<:Number, 1}}```
"""
NodalField{NF, NN, Vector, T}(::UndefInitializer)  where {NF, NN, T} = VectorizedNodalField{NF, NN, T}(undef)
"""
```NodalField{NF, NN, Vector, T}(::UndefInitializer)  where {NF, NN, T}```
"""
NodalField{NF, NN, Matrix, T}(::UndefInitializer)  where {NF, NN, T <: Number} = SimpleNodalField{NF, NN, T}(undef)
"""
```NodalField{NF, NN, Matrix, T}(::UndefInitializer)  where {NF, NN, T <: Number}```
"""
NodalField{NF, NN, StructArray, T}(::UndefInitializer) where {NF, NN, T} = VectorizedNodalField{NF, NN, StructArray, T}(undef)
"""
```NodalField{NF, NN, StructArray, T}(::UndefInitializer) where {NF, NN, T}```
"""
NodalField{Tup, A, T}(::UndefInitializer) where {Tup, A, T} = NodalField{Tup[1], Tup[2], A, T}(undef)
"""
```NodalField{Tup, A, T}(::UndefInitializer) where {Tup, A, T}```
"""
NodalField{Tup, A}(vals::M) where {Tup, A, M <: AbstractArray} = NodalField{Tup[1], Tup[2], A}(vals)

##########################################################################################

"""
```ElementField{NN, NE, Vector}(vals::M) where {NN, NE, M <: AbstractArray{<:Number, 2}}```
"""
ElementField{NN, NE, Vector}(vals::M) where {NN, NE, M <: AbstractArray{<:Number, 2}}    = VectorizedElementField{NN, NE}(vals)
"""
```ElementField{NN, NE, Matrix}(vals::M) where {NN, NE, M <: AbstractArray{<:Number, 2}}```
"""
ElementField{NN, NE, Matrix}(vals::M) where {NN, NE, M <: AbstractArray{<:Number, 2}}    = SimpleElementField{NN, NE}(vals)
"""
```ElementField{NN, NE, Vector}(vals::V) where {NN, NE, V <: AbstractArray{<:Number, 1}}```
"""
ElementField{NN, NE, Vector}(vals::V) where {NN, NE, V <: AbstractArray{<:Number, 1}}    = VectorizedElementField{NN, NE}(vals)
"""
```ElementField{NN, NE, Vector, T}(::UndefInitializer)  where {NN, NE, T}```
"""
ElementField{NN, NE, Vector, T}(::UndefInitializer)  where {NN, NE, T} = VectorizedElementField{NN, NE, T}(undef)
"""
```ElementField{NN, NE, Matrix, T}(::UndefInitializer)  where {NN, NE, T <: Number}```
"""
ElementField{NN, NE, Matrix, T}(::UndefInitializer)  where {NN, NE, T <: Number} = SimpleElementField{NN, NE, Matrix, T}(undef)
"""
```ElementField{NN, NE, StructArray, T}(::UndefInitializer) where {NN, NE, T}```
"""
ElementField{NN, NE, StructArray, T}(::UndefInitializer) where {NN, NE, T} = VectorizedElementField{NN, NE, StructArray, T}(undef)
"""
```ElementField{Tup, A, T}(::UndefInitializer) where {Tup, A, T}```
"""
ElementField{Tup, A, T}(::UndefInitializer) where {Tup, A, T} = ElementField{Tup[1], Tup[2], A, T}(undef)
"""
```ElementField{Tup, A}(vals::M) where {Tup, A, M <: AbstractArray}```
"""
ElementField{Tup, A}(vals::M) where {Tup, A, M <: AbstractArray} = ElementField{Tup[1], Tup[2], A}(vals)

###############################################################################
"""
```QuadratureField{NF, NQ, NE, Matrix}(vals::Matrix{<:Number})      where {NF, NQ, NE}```
"""
QuadratureField{NF, NQ, NE, Matrix}(vals::Matrix{<:Number})      where {NF, NQ, NE}    = SimpleQuadratureField{NF, NQ, NE}(vals)
"""
```QuadratureField{NF, NQ, NE, Vector}(vals::Matrix{<:Number})      where {NF, NQ, NE}```
"""
QuadratureField{NF, NQ, NE, Vector}(vals::Matrix{<:Number})      where {NF, NQ, NE}    = VectorizedQuadratureField{NF, NQ, NE}(vals)
"""
```QuadratureField{NF, NQ, NE, Matrix, T}(::UndefInitializer)       where {NF, NQ, NE, T}```
"""
QuadratureField{NF, NQ, NE, Matrix, T}(::UndefInitializer)       where {NF, NQ, NE, T} = SimpleQuadratureField{NF, NQ, NE, Matrix, T}(undef)
"""
```QuadratureField{NF, NQ, NE, StructArray, T}(::UndefInitializer)  where {NF, NQ, NE, T}```
"""
QuadratureField{NF, NQ, NE, StructArray, T}(::UndefInitializer)  where {NF, NQ, NE, T} = SimpleQuadratureField{NF, NQ, NE, StructArray, T}(undef)
"""
```QuadratureField{NF, NQ, NE, StructVector, T}(::UndefInitializer) where {NF, NQ, NE, T}```
"""
QuadratureField{NF, NQ, NE, StructVector, T}(::UndefInitializer) where {NF, NQ, NE, T} = VectorizedQuadratureField{NF, NQ, NE, StructVector, T}(undef)
"""
```QuadratureField{NF, NQ, NE, Vector, T}(::UndefInitializer)       where {NF, NQ, NE, T}```
"""
QuadratureField{NF, NQ, NE, Vector, T}(::UndefInitializer)       where {NF, NQ, NE, T} = VectorizedQuadratureField{NF, NQ, NE, Vector, T}(undef)
"""
```QuadratureField{Tup, A, T}(::UndefInitializer) where {Tup, A, T}```
"""
QuadratureField{Tup, A, T}(::UndefInitializer) where {Tup, A, T} = QuadratureField{Tup[1], Tup[2], Tup[3], A, T}(undef)
"""
```QuadratureField{Tup, A}(vals::M) where {Tup, A, M <: AbstractArray}```
"""
QuadratureField{Tup, A}(vals::M) where {Tup, A, M <: AbstractArray} = QuadratureField{Tup[1], Tup[2], Tup[3], A}(vals) 
