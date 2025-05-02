# abstract type AbstractPreconditioner end
abstract type AbstractSolver end

abstract type AbstractLinearSolver{
  A <: AbstractAssembler, 
  P, 
  U <: AbstractArray{<:Number, 1}
} <: AbstractSolver end
abstract type AbstractNonLinearSolver{
  L <: AbstractLinearSolver
} <: AbstractSolver end

# traditional solver interface
function preconditioner end
function residual end
function solve! end
function tangent end
function update_preconditioner! end

include("DirectLinearSolver.jl")
include("IterativeLinearSolver.jl")
include("NewtonSolver.jl")
