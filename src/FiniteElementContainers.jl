module FiniteElementContainers

# types
export Assembler
export AssemblerCache
export DofManager
export EssentialBC
export FunctionSpace
export FunctionSpaceInterpolant
export Mesh

# methods
export assemble!
export assemble_residual!
export create_fields
export create_unknowns
export reset!
export update_bc!
export update_bcs!
export update_fields!
export update_scratch!

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

end
