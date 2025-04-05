abstract type AbstractPreconditioner end
abstract type AbstractSolver end

abstract type AbstractLinearSolver{A <: AbstractAssembler, P <: AbstractPreconditioner} <: AbstractSolver end
abstract type AbstractNonLinearSolver{L <: AbstractLinearSolver} <: AbstractSolver end

# traditional solver interface
function preconditioner end
function residual end
function tangent end
function update_preconditioner! end

# optimization based solver interface
function gradient end
function hessian end
function value end

function solve! end

struct DenseLinearSolver{Inc <: AbstractArray{<:Number, 1}} <: AbstractLinearSolver{<:SparseMatrixAssembler, P}
  assembler::A
  preconditioner::P
  # TODO add some tolerances
  # what's the best way to do this with general solvers?
  ΔUu::Inc
end

function DenseLinearSolver(assembler::SparseMatrixAssembler)
  preconditioner = I
  ΔUu = similar(assembler.residual_unknowns)
  fill!(ΔUu, zero(eltype(ΔUu)))
  return DenseLinearSolver(assembler, preconditioner, ΔUu)
end 

function solve!(solver::DenseLinearSolver, Uu, p)

end
