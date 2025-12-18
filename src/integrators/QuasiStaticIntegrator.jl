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
  KA.synchronize(KA.get_backend(integrator))
  # the call below will ensure the return fields have bcs properly enfroced
  # before being saved as the old solution for the next step.
  _update_for_assembly!(p, integrator.solver.linear_solver.assembler.dof, integrator.solution)
  p.h1_field_old.data .= p.h1_field.data
  return nothing
end
