module FiniteElementContainers

# types
export DofManager
export EssentialBC
export FunctionSpace
export LinearSystem
export Mesh

# methods
export create_fields
export create_unknowns
export update_bc!
export update_bcs!
export update_fields!

using Exodus
using LinearAlgebra
using LoopVectorization
using ReferenceFiniteElements
using SparseArrays
using StaticArrays
using StructArrays

const MeshTypes = Union{
  ExodusDatabase
}

abstract type AbstractFEMContainer end
abstract type AbstractCellContainer <: AbstractFEMContainer end

include("Mesh.jl")

include("EssentialBCs.jl")
include("FunctionSpaces.jl")
# include("Variables.jl")
# include("FunctionSpaces.jl")

include("DofManagers.jl")
include("LinearSystems.jl")

end
