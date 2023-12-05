module FiniteElementContainers

# Meshes exports
export FileMesh
export Mesh
export coordinates
export element_block_ids
export element_connectivity
export element_type
export nodeset
export nodeset_ids
export num_dimensions
export num_nodes
export sideset
export sideset_ids

# Fields
export ElementField
export NodalField
export QuadratureField

# Connectivities
export Connectivity

# Methods
export connectivity
export element_level_coordinates
export element_level_coordinates!
export element_level_fields
export element_level_fields!

# dependencies
using DocStringExtensions
using LinearAlgebra
using ReferenceFiniteElements
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

include("Fields.jl")
include("Connectivities.jl")

include("Meshes.jl")


include("FunctionSpaces.jl")

end # module