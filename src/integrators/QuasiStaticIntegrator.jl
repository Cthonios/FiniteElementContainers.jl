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
  @show p.times.time_current
  @info "Current Time = $(current_time(p.times))"
  # update_bcs!(H1Field, integrator.solver, p)
  update_bcs!(integrator, p)
  solve!(integrator.solver, integrator.solution, p)
  return nothing
end

function update_bcs!(::QuasiStaticIntegrator, p::Parameters)
  X = p.h1_coords # TODO won't work for mixed bcs
  @show t = current_time(p.times)

  update_bc_values!(p.dirichlet_bcs, X, t)

  for bc in values(p.dirichlet_bcs)
    _update_bcs!(bc, p.h1_field, KA.get_backend(bc))
  end
  return nothing
end