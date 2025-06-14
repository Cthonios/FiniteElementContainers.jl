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
  # ΔUu = KA.zeros(KA.get_backend)
  ΔUu = similar(assembler.residual_unknowns)
  fill!(ΔUu, zero(eltype(ΔUu)))
  KA.synchronize(KA.get_backend(ΔUu))
  n = length(ΔUu)
  solver = eval(solver_sym)(n, n, typeof(ΔUu))
  return IterativeLinearSolver(assembler, preconditioner, solver, TimerOutput(), ΔUu)
end

# TODO specialize for operator like assemblers
function solve!(solver::IterativeLinearSolver, Uu, p)
  # assemble relevant fields
  @timeit solver.timer "residual assembly" begin
    assemble!(solver.assembler, Uu, p, Val{:residual}(), H1Field)
    # assemble_vector_neumann_bc!(solver.assembler, Uu, p, H1Field)
  end
  @timeit solver.timer "stiffness assembly" begin
    assemble!(solver.assembler, Uu, p, Val{:stiffness}(), H1Field)
  end
  # solve and fetch solution
  @timeit solver.timer "solve" begin
    Krylov.solve!(solver.solver, stiffness(solver.assembler), residual(solver.assembler))
    KA.synchronize(KA.get_backend(solver))
  end
  @timeit solver.timer "update solution" begin
    ΔUu = -Krylov.solution(solver.solver)
    # make necessary copies and updates
    copyto!(solver.ΔUu, ΔUu)
    KA.synchronize(KA.get_backend(solver))
    map!((x, y) -> x + y, Uu, Uu, ΔUu)
  end
  return nothing
end
