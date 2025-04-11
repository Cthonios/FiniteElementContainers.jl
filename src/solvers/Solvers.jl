# abstract type AbstractPreconditioner end
abstract type AbstractSolver end

abstract type AbstractLinearSolver{A <: AbstractAssembler, P} <: AbstractSolver end
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

struct DirectLinearSolver{
  A <: SparseMatrixAssembler, 
  P, 
  Inc <: AbstractArray{<:Number, 1}
} <: AbstractLinearSolver{A, P}
  assembler::A
  preconditioner::P
  # TODO add some tolerances
  # what's the best way to do this with general solvers?
  ΔUu::Inc
end

function DirectLinearSolver(assembler::SparseMatrixAssembler)
  preconditioner = I
  ΔUu = similar(assembler.residual_unknowns)
  fill!(ΔUu, zero(eltype(ΔUu)))
  return DirectLinearSolver(assembler, preconditioner, ΔUu)
end 

function solve!(solver::DirectLinearSolver, Uu, p)
  assemble!(solver.assembler, p.physics, p.h1_field, :residual_and_stiffness)
  R = residual(solver.assembler)
  K = stiffness(solver.assembler)
  # TODO make a KA kernel for a copy here
  # TODO specialize to backend solvers if they exists
  solver.ΔUu .= -K \ R
  update_field_unknowns!(p.h1_field, solver.assembler.dof, solver.ΔUu, +)
  return nothing
end

struct NewtonSolver{L, T} <: AbstractNonLinearSolver{L}
  linear_solver::L
  #
  max_iters::Int
  abs_increment_tol::T
  abs_residual_tol::T
end

function NewtonSolver(linear_solver)
  return NewtonSolver(linear_solver, 10, 1e-12, 1e-12)
end

function solve!(solver::NewtonSolver, Uu, p)
  for n in 1:solver.max_iters
    solve!(solver.linear_solver, Uu, p)
    @show norm(solver.linear_solver.ΔUu) norm(residual(solver.linear_solver.assembler))
    if norm(solver.linear_solver.ΔUu) < 1e-12 || norm(residual(solver.linear_solver.assembler)) < 1e-12
      break
    end
  end
end
