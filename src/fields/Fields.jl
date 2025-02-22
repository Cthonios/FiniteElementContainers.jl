"""
$(TYPEDEF)
Thin wrapper that subtypes ```AbstractArray``` and serves
as the base ```Field``` type
"""
abstract type AbstractField{T, N, NF, Vals} <: AbstractArray{T, N} end

function KA.get_backend(field::AbstractField)
  return KA.get_backend(field.vals)
end

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
num_elements(::QuadratureField{T, N, NF, NQ, NE, Vals}) where {T, N, NF, NQ, NE, Vals} = NE

"""
$(TYPEDSIGNATURES)
"""
num_q_points(::QuadratureField{T, N, NF, NQ, NE, Vals}) where {T, N, NF, NQ, NE, Vals} = NQ

###############################################
# implementations
# include("ComponentArrayElementField.jl")
include("ComponentArrayQuadratureField.jl")
# include("SimpleElementField.jl")
# include("SimpleNodalField.jl")
include("SimpleQuadratureField.jl")
# include("VectorizedElementField.jl")
# include("VectorizedNodalField.jl")
include("VectorizedQuadratureField.jl")
#

include("ElementField.jl")
include("NodalField.jl")

##########################################################################################

# ElementField{NN, NE, Vector}(vals::M) where {NN, NE, M <: AbstractArray{<:Number, 2}}    = VectorizedElementField{NN, NE}(vals)
# ElementField{NN, NE, Matrix}(vals::M) where {NN, NE, M <: AbstractArray{<:Number, 2}}    = SimpleElementField{NN, NE}(vals)
# ElementField{NN, NE, Vector}(vals::V) where {NN, NE, V <: AbstractArray{<:Number, 1}}    = VectorizedElementField{NN, NE}(vals)
# ElementField{NN, NE, Vector, T}(::UndefInitializer)  where {NN, NE, T} = VectorizedElementField{NN, NE, T}(undef)
# ElementField{NN, NE, Matrix, T}(::UndefInitializer)  where {NN, NE, T <: Number} = SimpleElementField{NN, NE, Matrix, T}(undef)
# ElementField{NN, NE, StructArray, T}(::UndefInitializer) where {NN, NE, T} = VectorizedElementField{NN, NE, StructArray, T}(undef)
# ElementField{Tup, A, T}(::UndefInitializer) where {Tup, A, T} = ElementField{Tup[1], Tup[2], A, T}(undef)
# ElementField{Tup, A}(vals::M) where {Tup, A, M <: AbstractArray} = ElementField{Tup[1], Tup[2], A}(vals)
# ElementField{NF, NE, ComponentArray, T}(::UndefInitializer, names) where {NF, NE, T} = ComponentArrayElementField{NF, NE, T}(undef, names)

###############################################################################

QuadratureField{NF, NQ, NE, Matrix}(vals::Matrix{<:Number})      where {NF, NQ, NE}    = SimpleQuadratureField{NF, NQ, NE}(vals)
QuadratureField{NF, NQ, NE, Vector}(vals::Matrix{<:Number})      where {NF, NQ, NE}    = VectorizedQuadratureField{NF, NQ, NE}(vals)
QuadratureField{NF, NQ, NE, Matrix, T}(::UndefInitializer)       where {NF, NQ, NE, T} = SimpleQuadratureField{NF, NQ, NE, Matrix, T}(undef)
QuadratureField{NF, NQ, NE, StructArray, T}(::UndefInitializer)  where {NF, NQ, NE, T} = SimpleQuadratureField{NF, NQ, NE, StructArray, T}(undef)
QuadratureField{NF, NQ, NE, StructVector, T}(::UndefInitializer) where {NF, NQ, NE, T} = VectorizedQuadratureField{NF, NQ, NE, StructVector, T}(undef)
QuadratureField{NF, NQ, NE, Vector, T}(::UndefInitializer)       where {NF, NQ, NE, T} = VectorizedQuadratureField{NF, NQ, NE, Vector, T}(undef)
QuadratureField{Tup, A, T}(::UndefInitializer) where {Tup, A, T} = QuadratureField{Tup[1], Tup[2], Tup[3], A, T}(undef)
QuadratureField{Tup, A}(vals::M) where {Tup, A, M <: AbstractArray} = QuadratureField{Tup[1], Tup[2], Tup[3], A}(vals) 
QuadratureField{NF, NQ, NE, ComponentArray, T}(::UndefInitializer, names) where {NF, NQ, NE, T} = ComponentArrayQuadratureField{NF, NQ, NE, T}(undef, names)
