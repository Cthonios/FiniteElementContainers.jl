"""
$(TYPEDSIGNATURES)
Assemble the diagonal of a matrix (e.g., stiffness or mass) into the
assembler's residual storage field.  Uses the same physics function as
`assemble_stiffness!` / `assemble_mass!` but extracts only the diagonal
entries of the element matrix at each quadrature point, avoiding full
sparse matrix assembly.

The result is stored in the assembler's residual storage and can be
retrieved via `diagonal(asm)`.

This is essential for GPU-friendly Jacobi preconditioning: `diag(K)`
gives the true diagonal, whereas the row-sum approximation `K·1` can
be zero at interior nodes of uniform meshes.
"""
function assemble_diagonal!(
  assembler, func::F, Uu, p
) where F <: Function
  storage = assembler.residual_storage
  fill!(storage, zero(eltype(storage)))
  dof = assembler.dof
  fspace = function_space(dof)
  X = coordinates(p)
  t = current_time(p)
  Δt = time_step(p)
  U = p.field
  U_old = p.field_old
  _update_for_assembly!(p, dof, Uu)
  return_type = AssembledDiagonal()
  conns = fspace.elem_conns
  foreach_block(fspace, p) do physics, props, ref_fe, b
    _assemble_block!(
      storage,
      conns.data, conns.offsets[b],
      func,
      physics, ref_fe,
      X, t, Δt,
      U, U_old,
      block_view(p.state_old, b), block_view(p.state_new, b), props,
      return_type
    )
  end
  return nothing
end

"""
$(TYPEDSIGNATURES)
Return the assembled diagonal vector after a call to `assemble_diagonal!`.
Extracts the unknown DOF subset (non-condensed) or the full vector (condensed).
"""
function diagonal(asm::AbstractAssembler)
  if _is_condensed(asm.dof)
    return asm.residual_storage.data
  else
    extract_field_unknowns!(
      asm.residual_unknowns,
      asm.dof,
      asm.residual_storage
    )
    return asm.residual_unknowns
  end
end
