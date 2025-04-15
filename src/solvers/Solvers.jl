# abstract type AbstractPreconditioner end
abstract type AbstractSolver end

abstract type AbstractLinearSolver{A <: AbstractAssembler, P, Inc} <: AbstractSolver end
abstract type AbstractNonLinearSolver{L <: AbstractLinearSolver} <: AbstractSolver end

# traditional solver interface
function preconditioner end
function residual end
function tangent end
function update_preconditioner! end

function update_bcs!(::Type{H1Field}, solver::AbstractLinearSolver, Uu, p)
  X = solver.assembler.dof.H1_vars[1].fspace.coords
  t = current_time(p.times)

  # rework this maybe so it's not resized?
  resize!(p.h1_dbcs, 0)

  for bc in values(p.dirichlet_bcs)
    update_bc_values!(bc, X, t)
    append!(p.h1_dbcs, bc.vals)
  end
  # remove me once you add error checking on dofs
  if length(p.h1_dbcs) != length(solver.assembler.dof.H1_bc_dofs)
    @warn "You may have a BC dof that is repeated. Beware!"
    resize!(p.h1_dbcs, length(solver.assembler.dof.H1_bc_dofs))
  end

  update_field!(p.h1_field, solver.assembler.dof, Uu, p.h1_dbcs)
  return nothing
end

function update_bcs!(::Type{H1Field}, solver::AbstractNonLinearSolver, Uu, p)
  update_bcs!(H1Field, solver.linear_solver, Uu, p)
  return nothing
end

# optimization based solver interface
function gradient end
function hessian end
function value end

function solve! end

struct DirectLinearSolver{
  A <: SparseMatrixAssembler, 
  P, 
  Inc <: AbstractArray{<:Number, 1}
} <: AbstractLinearSolver{A, P, Inc}
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
  # solver.ΔUu .= -K \ R
  copyto!(solver.ΔUu, -K \ R)
  update_field_unknowns!(p.h1_field, solver.assembler.dof, solver.ΔUu, +)
  return nothing
end

struct IterativeSolver{
  A,
  P,
  Inc,
  S
} <: AbstractLinearSolver{A, P, Inc}
  assembler::A
  preconditioner::P
  ΔUu::Inc
  solver::S
end

function IterativeSolver(assembler::SparseMatrixAssembler, solver_sym)
  # TODO
  preconditioner = I
  # ΔUu = KA.zeros(KA.get_backend)
  ΔUu = similar(assembler.residual_unknowns)
  fill!(ΔUu, zero(eltype(ΔUu)))
  n = length(ΔUu)
  solver = eval(solver_sym)(n, n, typeof(ΔUu))
  return IterativeSolver(assembler, preconditioner, ΔUu, solver)
end

# TODO specialize for operator like assemblers
function solve!(solver::IterativeSolver, Uu, p)
  # assemble!(solver.assembler, p.physics, p.h1_field, :residual_and_stiffness)
  assemble!(solver.assembler, p.physics, p.h1_field, :residual)
  assemble!(solver.assembler, p.physics, p.h1_field, :stiffness)
  Krylov.solve!(solver.solver, stiffness(solver.assembler), residual(solver.assembler))
  ΔUu = -Krylov.solution(solver.solver)
  # solver.ΔUu .= ΔUu 
  copyto!(solver.ΔUu, ΔUu)
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
