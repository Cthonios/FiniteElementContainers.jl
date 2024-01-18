module FiniteElementContainers

# Meshes exports
export FileMesh
export coordinates
export element_block_ids
export element_connectivity
export element_type
export nodeset
export nodeset_ids
export num_dimensions
export num_dofs_per_node
export num_fields
export num_nodes
export num_nodes_per_element
export num_q_points
export sideset
export sideset_ids

# Fields
export ElementField
export NodalField
export QuadratureField
# export field_names

# Connectivities
export Connectivity

# Methods
export connectivity
export element_level_fields
# export element_level_fields!

# DofManager
export DofManager
export create_fields
export create_unknowns
export dof_connectivity
export update_fields!
export update_unknown_dofs!

# FunctionSpaces
# export AbstractFunctionSpace
export AbstractMechanicsFormulation
export FunctionSpace
export NonAllocatedFunctionSpace
export VectorizedPreAllocatedFunctionSpace
export num_elements
export num_q_points

# Assemblers
export Assembler
export DynamicAssembler
export StaticAssembler
export assemble!

# dependencies
using DocStringExtensions
using LinearAlgebra
using ReferenceFiniteElements
using SparseArrays
using StaticArrays
using StructArrays
using Tensors

# for docs
@template (FUNCTIONS, METHODS, MACROS) =
"""
$(TYPEDSIGNATURES)
$(DOCSTRING)
$(METHODLIST)
"""

@template (TYPES) =
"""
$(TYPEDFIELDS)
$(DOCSTRING)
"""

abstract type FEMContainer end

include("fields/Fields.jl")
include("Connectivities.jl")
include("Meshes.jl")

include("DofManagers.jl")
include("FunctionSpaces.jl")
# include("Assemblers.jl")
include("assemblers/Assemblers.jl")

end # module