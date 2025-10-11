module FiniteElementContainers

# general
export cpu
export cuda
export rocm

# Assemblers
export SparseMatrixAssembler
export assemble_mass!
export assemble_matrix!
export assemble_matrix_action!
export assemble_scalar!
export assemble_stiffness!
export assemble_vector!
export assemble_vector_neumann_bc!

# BCs
export DirichletBC
export DirichletBCs
export NeumannBC
export NeumannBCs
export update_field_dirichlet_bcs!

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

# ICs
export InitialCondition
export InitialConditions
export update_field_ics!
export update_ic_values!

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
export num_fields
export num_nodes
export num_nodes_per_element
export sideset
export sidesets
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
export hvp
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
using Adapt
using Atomix
using DocStringExtensions
using ForwardDiff
using Krylov
using LinearAlgebra
using MPI
using ReferenceFiniteElements
using SparseArrays
using StaticArrays
using Tensors
using TimerOutputs

# hooks for extensions
function cpu end
function cuda end
function rocm end

function communication_graph end
function decompose_mesh end
function global_colorings end

# TODO need to further specialize for staticarrays, etc.
cpu(x) = adapt(Array, x)

# TODO clean this up, make it make sense in an ordered way
# include("parallel/Parallel.jl")

include("fields/Fields.jl")
include("Meshes.jl")
include("FunctionSpaces.jl")
include("Functions.jl")
include("DofManagers.jl")
include("bcs/BoundaryConditions.jl")
include("ics/InitialConditions.jl")

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
