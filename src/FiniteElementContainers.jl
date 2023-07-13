module FiniteElementContainers

# types
export EssentialBC
export FunctionSpace
export Mesh

# methods
export connectivity
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

include("EssentialBCs.jl")
include("FunctionSpaces.jl")
include("Mesh.jl")

include("DofManagers.jl")

end
