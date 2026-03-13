struct IterativeLinearSolver{
  A <: AbstractAssembler,
  P,
  S,
  T <: TimerOutput,
  U <: AbstractArray{<:Number, 1}
} <: AbstractLinearSolver{A, P, T, U}
  assembler::A
  preconditioner::P
  solver::S
  timer::T
  ΔUu::U
end

function IterativeLinearSolver(assembler::SparseMatrixAssembler, solver_sym)
  # TODO
  preconditioner = I
  ΔUu = similar(assembler.residual_unknowns)
  fill!(ΔUu, zero(eltype(ΔUu)))
  solver = krylov_workspace(Val(solver_sym), stiffness(assembler), residual(assembler))
  return IterativeLinearSolver(assembler, preconditioner, solver, TimerOutput(), ΔUu)
end

# TODO specialize for operator like assemblers
function solve!(solver::IterativeLinearSolver, Uu, p)
  # assemble relevant fields
  @timeit solver.timer "residual assembly" begin
    assemble_vector!(solver.assembler, residual, Uu, p)
    assemble_vector_neumann_bc!(solver.assembler, Uu, p)
  end
  @timeit solver.timer "stiffness assembly" begin
    assemble_stiffness!(solver.assembler, stiffness, Uu, p)
  end
  # solve and fetch solution
  @timeit solver.timer "solve" begin
    krylov_solve!(solver.solver, stiffness(solver.assembler), residual(solver.assembler))
  end
  @timeit solver.timer "update solution" begin
    ΔUu = -Krylov.solution(solver.solver)
    # make necessary copies and updates
    copyto!(solver.ΔUu, ΔUu)
    map!((x, y) -> x + y, Uu, Uu, ΔUu)
  end
  return nothing
end
