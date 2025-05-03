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
