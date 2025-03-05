module FiniteElementContainers

# TODO clean up exports

# Assemblers
# export Assembler
# export DynamicAssembler
# export MatrixFreeStaticAssembler
export SparseMatrixAssembler
# export StaticAssembler
export assemble!

# BCs
export DirichletBC
export NeumannBC

# Connectivities
export Connectivity

# Fields
export H1Field
export L2ElementField
# export QuadratureField
# export field_names

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

# Methods
export connectivity
export element_level_fields
# export element_level_fields!

# DofManager
export DofManager
export NewDofManager
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

# FunctionSpaces
export AbstractMechanicsFormulation
export FunctionSpace
export NonAllocatedFunctionSpace
export VectorizedPreAllocatedFunctionSpace

# TODO eventually remove these
export H1
export L2Element

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

# dependencies
import AcceleratedKernels as AK
import KernelAbstractions as KA
import Reexport: @reexport
using Atomix
using DocStringExtensions
@reexport using LinearAlgebra
@reexport using ReferenceFiniteElements
@reexport using SparseArrays
using StaticArrays
using StructArrays
using Tensors

abstract type FEMContainer end

# basic stuff
include("fields/Fields.jl")
include("Meshes.jl")

# clean this up
include("function_spaces/Utils.jl")


include("NewFunctionSpaces.jl")
include("Functions.jl")

#
include("NewDofManagers.jl")

#
include("bcs/BoundaryConditions.jl")
include("formulations/Formulations.jl")
# # include("assemblers/Assemblers.jl")

#
include("NewAssemblers.jl")

end # module
