struct DirectLinearSolver{
  A <: SparseMatrixAssembler, 
  P, 
  Inc <: AbstractArray{<:Number, 1}
} <: AbstractLinearSolver{A, P, Inc}
  assembler::A
  preconditioner::P
  # TODO add some tolerances
  # what's the best way to do this with general solvers?
  ΔUu::Inc
end

function DirectLinearSolver(assembler::SparseMatrixAssembler)
  preconditioner = I
  ΔUu = similar(assembler.residual_unknowns)
  fill!(ΔUu, zero(eltype(ΔUu)))
  return DirectLinearSolver(assembler, preconditioner, ΔUu)
end 

function solve!(solver::DirectLinearSolver, Uu, p)
  update_field_unknowns!(p.h1_field, solver.assembler.dof, Uu)
  assemble!(solver.assembler, H1Field, p, :residual_and_stiffness)
  R = residual(solver.assembler)
  K = stiffness(solver.assembler)
  # TODO specialize to backend solvers if they exists
  # solver.ΔUu .= -K \ R
  copyto!(solver.ΔUu, -K \ R)
  # update_field_unknowns!(p.h1_field, solver.assembler.dof, solver.ΔUu, +)
  map!((x, y) -> x + y, Uu, Uu, solver.ΔUu)
  return nothing
end
