abstract type AbstractIntegrator{Solver} end
# abstract type AbstractFirstOrderIntegrator <: AbstractIntegrator end
# abstract type AbstractSecondOrderIntegrator <: AbstractIntegrator end
abstract type AbstractStaticIntegrator{Solver, Solution} <: AbstractIntegrator{Solver, Solution} end

include("StaticIntegrator.jl")
