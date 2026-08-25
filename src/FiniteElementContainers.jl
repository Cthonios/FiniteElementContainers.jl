module FiniteElementContainers

# general
export cpu
export cuda
export rocm
export to_backend

# Assemblers
export SparseMatrixAssembler
export as_matrix_free
export create_assembler_cache
export assemble_lumped_mass!
export assemble_mass!
export assemble_matrix!
export assemble_matrix_action!
export assemble_matrix_free_action!
export assemble_matrix_free_action_full!
export assemble_scalar!
export assemble_stiffness!
export assemble_vector!
export assemble_vector_neumann_bc!
export assemble_vector_source!

# BCs
export DirichletBC
export DirichletBCs
export NeumannBC
export NeumannBCs
export PeriodicBC
export PeriodicBCs
export RobinBC
export RobinBCs
export Source
export Sources
export dirichlet_dofs
export update_bc_values!
export update_field_dirichlet_bcs!

# Connectivities
export Connectivity
export connectivity

# DofManager
export DofManager
export create_field
export create_unknowns
export update_dofs!
export update_field_unknowns!

# Fields
export H1Field
export HcurlField
export HdivField
export L2Field
export StateVariableField
export num_entities
export num_fields

# Formulations
export AbstractElementFormulation
export AbstractMechanicsElementFormulation
export GeneralFormulation
export PlaneStrain
export ThreeDimensional
export discrete_gradient
export discrete_symmetric_gradient
export discrete_values
export extract_stress
export extract_stiffness
export modify_field_gradients
export scatter_with_gradients!
export scatter_with_gradients_and_gradients!
export scatter_with_symmetric_gradients!
export scatter_with_values!
export scatter_with_values_and_values!

# FunctionSpaces
export FunctionSpace

export num_elements
# export num_q_points

# Functions
export GeneralFunction
export ScalarFunction
export SymmetricTensorFunction
export TensorFunction
export VectorFunction

# ICs
export InitialCondition
export InitialConditions
export update_field_ics!
export update_ic_values!

# Integrals
# export MatrixIntegral
# export ScalarIntegral
# export VectorIntegral
# export integrate
# export remove_fixed_dofs!

# Integrators
export AbstractIntegrator
export QuasiStaticIntegrator
export evolve!

# Meshes
export FileMesh
export StructuredMesh
export UnstructuredMesh
export block_names
export distribute_mesh
export element_blocks
export element_ids
export global_colorings
export nodal_coordinates
export nodesets # rename to boundary_nodes
export num_dimensions
export num_nodes
export sidesets # rename to boundary_facets or something like that

# Parameters
export Parameters
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
export unpack_field

export assemble_diagonal!
export diagonal
export energy
export hvp
export lumped_mass
export mass
export mass!
export mass_action
export mass_action!
export residual
export residual!
export stiffness
export stiffness!
export stiffness_action
export stiffness_action!

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

# Times
export TimeStepper
export current_time

# other exports from deps
export Lagrange
export MappedH1OrL2Interpolants
export MappedH1OrL2SurfaceInterpolants

# dependencies
import AcceleratedKernels as AK
import KernelAbstractions as KA
using Adapt
using Atomix
using BlockArrays
using DocStringExtensions
using Exodus
using ForwardDiff
using GPUArrays
using Krylov
using LinearAlgebra
using Preferences
using ReferenceFiniteElements
using SparseArrays
using SparseMatricesCSR
using StaticArrays
using Tensors
using TimerOutputs

##################
# preferences
##################
const ASSEMBLE_DIAGONAL_GPU_BLOCK_SIZE = @load_preference("assemble_diagonal_gpu_block_size", 256)
const ASSEMBLE_LUMPED_MASS_GPU_BLOCK_SIZE = @load_preference("assemble_lumped_mass_gpu_block_size", 256)
const ASSEMBLE_MATRIX_GPU_BLOCK_SIZE = @load_preference("assemble_matrix_gpu_block_size", 256)
const ASSEMBLE_MATRIX_ACTION_GPU_BLOCK_SIZE = @load_preference("assemble_matrix_action_gpu_block_size", 256)
const ASSEMBLE_MATRIX_FREE_ACTION_GPU_BLOCK_SIZE = @load_preference("assemble_matrix_free_action_gpu_block_size", 256)
const ASSEMBLE_MATRIX_WEAKLY_ENFORCED_BC_GPU_BLOCK_SIZE = @load_preference("assemble_matrix_weakly_enforced_bc_gpu_block_size", 256)
const ASSEMBLE_QUADRATURE_QUANTITY_GPU_BLOCK_SIZE = @load_preference("assemble_quadrature_quantity_gpu_block_size", 256)
const ASSEMBLE_VECTOR_GPU_BLOCK_SIZE = @load_preference("assemble_vector_gpu_block_size", 256)
const ASSEMBLE_VECTOR_SOURCE_GPU_BLOCK_SIZE = @load_preference("assemble_vector_source_gpu_block_size", 256)
const ASSEMBLE_VECTOR_WEAKLY_ENFORCE_BC_GPU_BLOCK_SIZE = @load_preference("assemble_vector_weakly_enforced_bc_gpu_block_size", 256)

function summarize_preferences()
    println("GPU Preferences:")
    println("  assemble_diagonal_gpu_block_size                  = ", ASSEMBLE_DIAGONAL_GPU_BLOCK_SIZE)
    println("  assemble_lumped_mass_gpu_block_size               = ", ASSEMBLE_LUMPED_MASS_GPU_BLOCK_SIZE)
    println("  assemble_matrix_gpu_block_size                    = ", ASSEMBLE_MATRIX_GPU_BLOCK_SIZE)
    println("  assemble_matrix_action_gpu_block_size             = ", ASSEMBLE_MATRIX_ACTION_GPU_BLOCK_SIZE)
    println("  assemble_matrix_free_action_gpu_block_size        = ", ASSEMBLE_MATRIX_FREE_ACTION_GPU_BLOCK_SIZE)
    println("  assemble_matrix_weakly_enforced_bc_gpu_block_size = ", ASSEMBLE_MATRIX_WEAKLY_ENFORCED_BC_GPU_BLOCK_SIZE)
    println("  assemble_quadrature_quantity_gpu_block_size       = ", ASSEMBLE_QUADRATURE_QUANTITY_GPU_BLOCK_SIZE)
    println("  assemble_vector_gpu_block_size                    = ", ASSEMBLE_VECTOR_GPU_BLOCK_SIZE)
    println("  assemble_vector_source_gpu_block_size             = ", ASSEMBLE_VECTOR_SOURCE_GPU_BLOCK_SIZE)
    println("  assemble_vector_weakly_enforced_bc_gpu_block_size = ", ASSEMBLE_VECTOR_WEAKLY_ENFORCE_BC_GPU_BLOCK_SIZE)
end
##################
# exceptions
##################
abstract type AbstractFECError <: Exception end
function Base.showerror(io::IO, e::AbstractFECError)
    println(io, e.msg)
end

# TODO clean this up, make it make sense in an ordered way
# include("parallel/Parallel.jl")
include("Expressions.jl")
include("Fields.jl")
include("PostProcessors.jl")
# include("fields/Fields.jl")
# include("Utils.jl")
include("meshes/Meshes.jl")
include("FunctionSpaces.jl")
include("Functions.jl")
include("DofManagers.jl")
# where is best to put this?
# include("Utils.jl")

include("bcs/BoundaryConditions.jl")
include("Constraints.jl")
include("InitialConditions.jl")

include("Formulations.jl")
include("Physics.jl")
include("assemblers/Assemblers.jl")
#
include("TimeSteppers.jl")
include("Parameters.jl")
include("integrals/Integrals.jl")
# include("solvers/Solvers.jl")
include("Solvers.jl")
include("integrators/Integrators.jl")
# TODO figure out how to better integrate this stuff
# maybe through a package extension?
include("Enzyme.jl")
include("Utils.jl")

# extras
include("parser/InputFileParser.jl")

include("AppTools.jl")

end # module
