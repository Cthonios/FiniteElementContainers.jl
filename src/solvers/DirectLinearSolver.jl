struct DirectLinearSolver{
  A <: SparseMatrixAssembler, 
  P, 
  T <: TimerOutput,
  Inc <: AbstractArray{<:Number, 1}
} <: AbstractLinearSolver{A, P, T, Inc}
  assembler::A
  preconditioner::P
  timer::T
  # TODO add some tolerances
  # what's the best way to do this with general solvers?
  ΔUu::Inc
end

function DirectLinearSolver(assembler::SparseMatrixAssembler)
  preconditioner = I
  ΔUu = similar(assembler.residual_unknowns)
  fill!(ΔUu, zero(eltype(ΔUu)))
  return DirectLinearSolver(assembler, preconditioner, TimerOutput(), ΔUu)
end 

function solve!(solver::DirectLinearSolver, Uu, p)
  assemble_vector!(solver.assembler, residual, Uu, p)
  assemble_vector_neumann_bc!(solver.assembler, Uu, p)
  assemble_stiffness!(solver.assembler, stiffness, Uu, p)
  R = residual(solver.assembler)
  K = stiffness(solver.assembler)
  # TODO specialize to backend solvers if they exists
  # solver.ΔUu .= -K \ R
  copyto!(solver.ΔUu, -K \ R) # currently doesn't work on GPU
  # update_field_unknowns!(p.h1_field, solver.assembler.dof, solver.ΔUu, +)
  map!((x, y) -> x + y, Uu, Uu, solver.ΔUu)
  return nothing
end
