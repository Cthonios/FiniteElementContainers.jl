struct NewmarkIntegrator{
  Solution,
  Solver
} <: AbstractSecondOrderIntegrator{Solution, Solver}
  solver::Solver
  U::Solution
  V::Solution
  A::Solution
  U_prev::Solution
  V_prev::Solution
  A_prev::Solution
  β::Float64
  γ::Float64
end

function NewmarkIntegrator(solver::AbstractNonLinearSolver, β::Float64, γ::Float64)
  U = similar(solver.linear_solver.ΔUu)
  fill!(U, zero(eltype(U)))
  
  V = copy(U)
  A = copy(U)
  U_prev = copy(U)
  V_prev = copy(U)
  A_prev = copy(U)

  return NewmarkIntegrator(solver, U, V, A, U_prev, V_prev, A_prev, β, γ)
end

function evolve!(integrator::NewmarkIntegrator, p)
  update_time!(p)
  update_bcs!(H1Field, integrator.solver, integrator.U, p)
  
end