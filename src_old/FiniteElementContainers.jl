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

# dependencies
# using CUDA
# using CUDA.CUDAKernels
using Exodus
using Graphs
using KernelAbstractions
using LinearAlgebra
using ReferenceFiniteElements
using SparseArrays
using StaticArrays
using StructArrays

const MeshTypes = Union{
  ExodusDatabase
}

# abstract type AbstractFEMContainer end
# abstract type AbstractCellContainer <: AbstractFEMContainer end

include("Meshes_new.jl")
include("FunctionSpaces_new.jl")
# include("Meshes.jl")
# include("EssentialBCs.jl")
# include("FunctionSpaces_old.jl") # just for the setup methods
# include("FunctionSpaces.jl")
# # include("FunctionSpaces_new.jl")
# include("DofManagers_new.jl")
# include("Assemblers_new.jl")

# include("ElementColorings.jl")

# include("GraphMeshes.jl")

# # GPU attempts
# include("gpu/Mesh.jl")


end
