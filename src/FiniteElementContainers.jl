module FiniteElementContainers

# general
export cpu
export cuda
export rocm

# Assemblers
export SparseMatrixAssembler
export assemble!
export assemble_matrix!
export assemble_matrix_action!
export assemble_scalar!
export assemble_vector!
export constraint_matrix

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
export create_field
export create_unknowns
export update_bcs!
export update_dofs!
export update_field_unknowns!

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
export element_block_id_map
export element_block_ids
export element_block_names
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
export create_initial_state
export create_properties
export interpolate_field_gradients
export interpolate_field_values
export interpolate_field_values_and_gradients
export map_interpolants
export num_properties
export num_states
export reshape_element_level_field

export energy
export mass
export residual
export stiffness

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
export MappedInterpolants
export MappedSurfaceInterpolants

# dependencies
import KernelAbstractions as KA
using Atomix
using DocStringExtensions
using Krylov
using LinearAlgebra
using ReferenceFiniteElements
using SparseArrays
using StaticArrays
using Tensors
using TimerOutputs

abstract type FEMContainer end

function cpu end
function cuda end
function rocm end

# TODO clean this up, make it make sense in an ordered way
include("fields/Fields.jl")
include("Meshes.jl")
include("FunctionSpaces.jl")
include("Functions.jl")
include("DofManagers.jl")
include("bcs/BoundaryConditions.jl")
include("formulations/Formulations.jl")
include("physics/Physics.jl")
include("assemblers/Assemblers.jl")
include("PostProcessors.jl")

#
include("TimeSteppers.jl")
include("Parameters.jl")
include("solvers/Solvers.jl")
include("integrators/Integrators.jl")

end # module
