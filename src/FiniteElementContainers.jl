module FiniteElementContainers

# general
export cpu
export gpu

# Assemblers
export SparseMatrixAssembler
export assemble!
export constraint_matrix
export residual
export stiffness

# BCs
export DirichletBC
export NeumannBC

# Connectivities
export Connectivity
export connectivity

# Fields
export H1Field
export L2ElementField
export L2QuadratureField

# DofManager
export DofManager
export create_bcs
export create_field
export create_unknowns
export update_bcs!
export update_dofs!
export update_field_bcs!
export update_field_unknowns!
export update_field!

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

export num_elements
# export num_q_points

# Functions
export ScalarFunction
export StateFunction
export SymmetricTensorFunction
export TensorFunction
export VectorFunction

# Integrators
export AbstractIntegrator
export QuasiStaticIntegrator
export evolve!

# Meshes
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

# Parameters
export Parameters
export TimeStepper
export create_parameters

# Physics
export AbstractPhysics
export num_properties
export num_states

# PostProcessors
export PostProcessor
export write_field
export write_times

# Solvers
# export AbstractPreconditioner
# export AbstractSolver
export DirectLinearSolver
export IterativeLinearSolver
export NewtonSolver
export solve!

# other exports from deps
export Lagrange

# dependencies
import AcceleratedKernels as AK
import KernelAbstractions as KA
using Atomix
using DocStringExtensions
using Krylov
using LinearAlgebra
using ReferenceFiniteElements
using SparseArrays
using StaticArrays
using Tensors

abstract type FEMContainer end

function cpu end
function gpu end

# TODO clean this up, make it make sense in an ordered way
include("fields/Fields.jl")
include("Meshes.jl")
include("FunctionSpaces.jl")
include("Functions.jl")
include("DofManagers.jl")
include("bcs/BoundaryConditions.jl")
include("formulations/Formulations.jl")
include("assemblers/Assemblers.jl")
include("physics/Physics.jl")
include("PostProcessors.jl")

#
include("Parameters.jl")
include("solvers/Solvers.jl")
include("integrators/Integrators.jl")

end # module
