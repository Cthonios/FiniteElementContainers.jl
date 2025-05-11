struct QuasiStaticIntegrator{
  Solution, 
  Solver
} <: AbstractStaticIntegrator{Solution, Solver}
  solver::Solver
  solution::Solution
end

function QuasiStaticIntegrator(solver::AbstractNonLinearSolver)
  solution = similar(solver.linear_solver.ΔUu)
  fill!(solution, zero(eltype(solution)))
  return QuasiStaticIntegrator(solver, solution)
end

function evolve!(integrator::QuasiStaticIntegrator, p)
  update_time!(p)
  @info "Current Time = $(current_time(p.times))"
  update_bc_values!(p)
  solve!(integrator.solver, integrator.solution, p)
  return nothing
end
