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

function KA.get_backend(integrator::AbstractIntegrator)
  return KA.get_backend(integrator.solution)
end

include("QuasiStaticIntegrator.jl")
