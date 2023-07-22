module FiniteElementContainers

# types
export DofManager
export EssentialBC
export FunctionSpace
export Mesh

# # methods
# export connectivity
# export create_field
export get_bc_size
export get_bc_values
export get_unknown_size
export get_unknown_values
# export JxWs
# export quadrature_point_coordinates
# export shape_function_gradients
# export shape_function_values

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
# include("Variables.jl")
# include("FunctionSpaces.jl")
include("Mesh.jl")

include("DofManagers.jl")

end
