# abstract type AbstractPreconditioner end
abstract type AbstractSolver end

abstract type AbstractLinearSolver{
  A <: AbstractAssembler, 
  P, 
  T <: TimerOutput,
  U <: AbstractArray{<:Number, 1}
} <: AbstractSolver end
abstract type AbstractNonLinearSolver{
  L <: AbstractLinearSolver,
  T <: TimerOutput
} <: AbstractSolver end

# traditional solver interface
function preconditioner end
function residual end
function solve! end
function tangent end
function update_preconditioner! end

function KA.get_backend(solver::AbstractLinearSolver)
  return KA.get_backend(solver.assembler)
end

function KA.get_backend(solver::AbstractNonLinearSolver)
  return KA.get_backend(solver.linear_solver)
end

include("DirectLinearSolver.jl")
include("IterativeLinearSolver.jl")
include("NewtonSolver.jl")
