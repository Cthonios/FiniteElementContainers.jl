module FiniteElementContainers

# types
export DofManager
export EssentialBC
export FunctionSpace
export Mesh
export StaticAssembler

# methods
export assemble!
export create_fields
export create_unknowns
export update_bcs!
export update_fields!

using Exodus
using LinearAlgebra
using ReferenceFiniteElements
using SparseArrays
using StaticArrays
using StructArrays

const MeshTypes = Union{
  ExodusDatabase
}

abstract type AbstractFEMContainer end
abstract type AbstractCellContainer <: AbstractFEMContainer end

include("Meshes.jl")
include("EssentialBCs.jl")
include("FunctionSpaces.jl")
include("DofManagers.jl")
include("Assemblers.jl")

include("ElementColorings.jl")

end
