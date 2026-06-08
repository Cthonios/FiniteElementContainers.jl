struct IterativeLinearSolver{
  A <: AbstractAssembler,
  P,
  U <: AbstractArray{<:Number, 1},
  W
} <: AbstractLinearSolver{A, P, U}
  assembler::A
  preconditioner::P
  timer::TimerOutput
  ΔUu::U
  workspace::W
end

function IterativeLinearSolver(assembler::SparseMatrixAssembler, solver_sym)
  # TODO
  preconditioner = I
  K = stiffness(assembler)
  R = residual(assembler)
  ΔUu = similar(R, axes(K, 1))
  fill!(ΔUu, zero(eltype(ΔUu)))
  workspace = krylov_workspace(Val(solver_sym), K, R)
  return IterativeLinearSolver(assembler, preconditioner, TimerOutput(), ΔUu, workspace)
end

# TODO specialize for operator like assemblers
function solve!(solver::IterativeLinearSolver, Uu, p)
  asm = solver.assembler
  if _use_inplace_methods(asm)
    residual_method = residual!
    stiffness_method = stiffness!
  else
    residual_method = residual
    stiffness_method = stiffness
  end
  # assemble relevant fields
  @timeit solver.timer "residual assembly" begin
    assemble_vector!(asm, residual_method, Uu, p)
    assemble_vector_source!(asm, Uu, p)
    assemble_vector_neumann_bc!(asm, Uu, p)
    # assemble_vector_robin_bc!(asm, Uu, p)
  end
  @timeit solver.timer "stiffness assembly" begin
    assemble_stiffness!(asm, stiffness_method, Uu, p)
  end
  # solve and fetch solution
  @timeit solver.timer "solve" begin
    krylov_solve!(solver.workspace, stiffness(asm), residual(asm))
  end
  @timeit solver.timer "update solution" begin
    ΔUu = -Krylov.solution(solver.workspace)
    # make necessary copies and updates
    copyto!(solver.ΔUu, ΔUu)
    map!((x, y) -> x + y, Uu, Uu, ΔUu)
  end
  return nothing
end
