module FiniteElementContainers

# types
export FunctionSpace

# methods
export connectivity
# export getindex
export JxWs
export quadrature_point_coordinates
export shape_function_gradients
export shape_function_values

using Exodus
using LinearAlgebra
using ReferenceFiniteElements
using StaticArrays
using StructArrays

const MeshTypes = Union{
  ExodusDatabase
}

abstract type AbstractFEMContainer end
abstract type AbstractCellContainer <: AbstractFEMContainer end

include("FunctionSpaces.jl")

end
