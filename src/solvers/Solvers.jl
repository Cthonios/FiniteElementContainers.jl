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

function _update_bcs!(::Type{H1Field}, bc, p, ::KA.CPU)
  for (dof, val) in zip(bc.bookkeeping.dofs, bc.vals)
    p.h1_field[dof] = val
  end
  return nothing
end

KA.@kernel function _update_bcs_kernel!(bc, p)
  I = KA.@index(Global)
  dof = bc.bookkeeping.dofs[I]
  val = bc.vals[I]
  p.h1_field[dof] = val
end

function _update_bcs!(::Type{H1Field}, bc, p, backend::KA.Backend)
  kernel! = _update_bcs_kernel!(backend)
  kernel!(bc, p, ndrange=length(bc.vals))
  return nothing
end

function update_bcs!(type::Type{H1Field}, solver::AbstractLinearSolver, Uu, p)
  X = solver.assembler.dof.H1_vars[1].fspace.coords
  t = current_time(p.times)

  update_bc_values!(p.dirichlet_bcs, X, t)
  # copyto!(p.h1_dbcs, p.dirichlet_bcs.vals)

  for bc in values(p.dirichlet_bcs)
    backend = KA.get_backend(bc)
    _update_bcs!(type, bc, p, backend)
  end

  # update_field!(p.h1_field, solver.assembler.dof, Uu, p.h1_dbcs)
  return nothing
end

function update_bcs!(::Type{H1Field}, solver::AbstractNonLinearSolver, Uu, p)
  update_bcs!(H1Field, solver.linear_solver, Uu, p)
  return nothing
end

include("DirectLinearSolver.jl")
include("IterativeLinearSolver.jl")
include("NewtonSolver.jl")
