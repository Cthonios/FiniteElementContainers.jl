struct IterativeLinearSolver{
  A <: AbstractAssembler,
  P,
  S,
  U <: AbstractArray{<:Number, 1}
} <: AbstractLinearSolver{A, P, U}
  assembler::A
  preconditioner::P
  solver::S
  ΔUu::U
end

function IterativeLinearSolver(assembler::SparseMatrixAssembler, solver_sym)
  # TODO
  preconditioner = I
  # ΔUu = KA.zeros(KA.get_backend)
  ΔUu = similar(assembler.residual_unknowns)
  fill!(ΔUu, zero(eltype(ΔUu)))
  n = length(ΔUu)
  solver = eval(solver_sym)(n, n, typeof(ΔUu))
  return IterativeLinearSolver(assembler, preconditioner, solver, ΔUu)
end

# TODO specialize for operator like assemblers
function solve!(solver::IterativeLinearSolver, Uu, p)
  # update unknown dofs
  update_field_unknowns!(p.h1_field, solver.assembler.dof, Uu)
  # assemble relevant fields
  assemble!(solver.assembler, H1Field, p, :residual)
  assemble!(solver.assembler, H1Field, p, :stiffness)
  # solve and fetch solution
  Krylov.solve!(solver.solver, stiffness(solver.assembler), residual(solver.assembler))
  ΔUu = -Krylov.solution(solver.solver)
  # make necessary copies and updates
  copyto!(solver.ΔUu, ΔUu)
  map!((x, y) -> x + y, Uu, Uu, ΔUu)
  return nothing
end
