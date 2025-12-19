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
# struct DirectLinearSolver{
#   U <: AbstractArray,
#   A <: MatrixIntegral,
#   b <: VectorIntegral,
#   P
# }
#   Δu::U
#   matrix::A
#   vector::b
#   precon::P
#   timer::TimerOutput
# end

# function DirectLinearSolver(A::MatrixIntegral, b::VectorIntegral)
#   precon = I
#   ΔUu = create_unknowns(A)
#   return DirectLinearSolver(Δu, A, b, precon, TimerOutput())
# end

# function solve!(solver::DirectLinearSolver, u, p)
#   R = solver.matrix(u, p)
#   K = solver.vector(u, p)
#   # TODO need bcs as well
#   solver.Δu .= -K \ R
#   map!((x, y) -> x + y, u, u, solver.Δu)
#   return nothing 
# end
