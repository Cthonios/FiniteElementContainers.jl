struct NewtonSolver{F, L, T, CB} <: AbstractNonLinearSolver{L, T}
  max_iters::Int
  abs_increment_tol::F
  abs_residual_tol::F
  rel_residual_tol::F
  linear_solver::L
  timer::T
  # Optional callback: (iter, norm_ΔUu, norm_R, rel_R, converged) -> nothing
  # Called after each Newton iteration. Set to `nothing` to disable.
  log_callback::CB
end

function NewtonSolver(linear_solver)
  return NewtonSolver(10, 1e-12, 1e-12, 1e-12, linear_solver, linear_solver.timer, nothing)
end

function solve!(solver::NewtonSolver, Uu, p)
  @timeit solver.timer "Nonlinear solver" begin
    initial_norm = Ref(0.0)
    for n in 1:solver.max_iters
      @timeit solver.timer "Linear solver" begin
        solve!(solver.linear_solver, Uu, p)
      end

      KA.synchronize(KA.get_backend(solver))

      @timeit solver.timer "convergence check" begin
        norm_ΔUu = sqrt(sum(abs2, solver.linear_solver.ΔUu))
        norm_R   = sqrt(sum(abs2, residual(solver.linear_solver.assembler)))
        if n == 1
          initial_norm[] = norm_R
        end
        rel_R     = initial_norm[] > 0.0 ? norm_R / initial_norm[] : norm_R
        converged = norm_ΔUu < solver.abs_increment_tol ||
                    norm_R   < solver.abs_residual_tol   ||
                    rel_R    < solver.rel_residual_tol
        @debug "Newton" n norm_ΔUu norm_R rel_R
        if !isnothing(solver.log_callback)
          solver.log_callback(n, norm_ΔUu, norm_R, rel_R, converged)
        end
        converged && break
      end

      KA.synchronize(KA.get_backend(solver))
    end
  end
end
