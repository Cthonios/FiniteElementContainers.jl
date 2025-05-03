abstract type AbstractIntegrator{
  Solution <: AbstractArray{<:Number, 1},
  Solver <: AbstractNonLinearSolver
} end
# abstract type AbstractFirstOrderIntegrator <: AbstractIntegrator end
# abstract type AbstractSecondOrderIntegrator <: AbstractIntegrator end
abstract type AbstractSecondOrderIntegrator{
  Solution,
  Solver
} <: AbstractIntegrator{Solution, Solver} end
abstract type AbstractStaticIntegrator{
  Solution,
  Solver
} <: AbstractIntegrator{Solution, Solver} end

function evolve! end

# TODO maybe move these methods below to BCs
function _update_bcs!(bc, U, ::KA.CPU)
  for (dof, val) in zip(bc.bookkeeping.dofs, bc.vals)
    U[dof] = val
  end
  return nothing
end

KA.@kernel function _update_bcs_kernel!(bc, U)
  I = KA.@index(Global)
  dof = bc.bookkeeping.dofs[I]
  val = bc.vals[I]
  U[dof] = val
end

function _update_bcs!(bc, U, backend::KA.Backend)
  kernel! = _update_bcs_kernel!(backend)
  kernel!(bc, U, ndrange=length(bc.vals))
  return nothing
end

include("QuasiStaticIntegrator.jl")
