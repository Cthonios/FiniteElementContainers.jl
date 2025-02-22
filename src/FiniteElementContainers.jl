module FiniteElementContainers

# Meshes exports
export FileMesh
export UnstructuredMesh
export coordinates
export element_block_ids
export element_connectivity
export element_type
export nodeset
export nodesets
export nodeset_ids
export nodeset_names
export num_dimensions
export num_dofs_per_node
export num_fields
export num_nodes
export num_nodes_per_element
export num_q_points
export sideset
export sideset_ids
export sideset_names

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
export create_field
export create_fields
export create_unknowns
export dof_connectivity
export update_dofs!
export update_field_bcs!
export update_field_unknowns!
export update_field!
export update_fields!
export update_unknown_dofs!

export NewDofManager

# FunctionSpaces
export AbstractMechanicsFormulation
export FunctionSpace
export NonAllocatedFunctionSpace
export VectorizedPreAllocatedFunctionSpace

export H1
export Hcurl
export Hdiv
export L2Element
export L2Quadrature

export element_level_coordinates
export element_level_fields
export num_elements
export num_q_points
export quadrature_level_field_gradients
export quadrature_level_field_values

export ScalarFunction
export SymmetricTensorFunction
export TensorFunction
export VectorFunction

# Assemblers
export Assembler
export DynamicAssembler
export MatrixFreeStaticAssembler
export SparseMatrixAssembler
export StaticAssembler
export assemble!

# Formulations
export IncompressiblePlaneStress
export PlaneStrain
export ScalarFormulation
export ThreeDimensional
export discrete_gradient
export discrete_symmetric_gradient
export discrete_values
export extract_stress
export extract_stiffness
export modify_field_gradients

# dependencies
import AcceleratedKernels as AK
import KernelAbstractions as KA
using Atomix
using ComponentArrays
using DocStringExtensions
using LinearAlgebra
using ReferenceFiniteElements
using SparseArrays
using StaticArrays
using StructArrays
using Tensors

abstract type FEMContainer end

include("fields/Fields.jl")
include("Connectivity.jl")
include("Meshes.jl")
include("DofManagers.jl")
include("function_spaces/NewFunctionSpaces.jl")
include("NewDofManagers.jl")

include("function_spaces/FunctionSpaces.jl")
# include("function_spaces/NewFunctionSpaces.jl")

include("formulations/Formulations.jl")
include("assemblers/Assemblers.jl")
include("NewAssemblers.jl")

end # module