struct NewtonSolver{F, L, T} <: AbstractNonLinearSolver{L, T}
  max_iters::Int
  abs_increment_tol::F
  abs_residual_tol::F
  linear_solver::L
  timer::T
end

# function NewtonSolver(linear_solver)
#   return NewtonSolver(10, 1e-12, 1e-12, linear_solver, TimerOutput())
# end

function NewtonSolver(linear_solver)
  return NewtonSolver(10, 1e-12, 1e-12, linear_solver, linear_solver.timer)
end

function solve!(solver::NewtonSolver, Uu, p)
  @timeit solver.timer "Nonlinear solver" begin
    for n in 1:solver.max_iters
      @timeit solver.timer "Linear solver" begin
        solve!(solver.linear_solver, Uu, p)
      end

      KA.synchronize(KA.get_backend(solver))

      @timeit solver.timer "convergence check" begin
        @show norm(solver.linear_solver.ΔUu) norm(residual(solver.linear_solver.assembler))
        if norm(solver.linear_solver.ΔUu) < 1e-12 || norm(residual(solver.linear_solver.assembler)) < 1e-12
          break
        end
      end

      KA.synchronize(KA.get_backend(solver))
    end
  end
end
