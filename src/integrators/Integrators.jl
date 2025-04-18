abstract type AbstractIntegrator{
  Solution <: AbstractArray{<:Number, 1},
  Solver <: AbstractNonLinearSolver
} end
# abstract type AbstractFirstOrderIntegrator <: AbstractIntegrator end
# abstract type AbstractSecondOrderIntegrator <: AbstractIntegrator end
abstract type AbstractStaticIntegrator{
  Solution,
  Solver
} <: AbstractIntegrator{Solution, Solver} end

function evolve! end

include("QuasiStaticIntegrator.jl")
