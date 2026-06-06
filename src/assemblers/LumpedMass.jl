"""
$(TYPEDSIGNATURES)
Assemble a partition-of-unity (row-sum) lumped mass vector.

`func` is a physics-level kernel with the same signature as `residual`
that returns an `SVector{NDOF, T}` of element-local lumped-mass
contributions per DOF.  For a partition-of-unity basis the contribution
is `density * N[a] * JxW`, identical in each spatial direction at node `a`.

The result is scattered into `assembler.residual_storage` (reused as
scratch — overwrites any previously-assembled residual) and retrieved
via `lumped_mass(assembler)`.

Distinct from:
  * `assemble_mass!`   — full consistent sparse mass matrix `M_ab = ρ ∫ N_a N_b dV`.
  * `assemble_diagonal!(asm, mass, ...)` — diagonal of the consistent
    matrix `M_aa = ρ ∫ N_a^2 dV`.  Not equal to the row-sum lumped mass
    in general (they differ by a factor depending on the element type).
  * `assemble_matrix_action!(asm, mass, U_zeros, ones_free, p)` —
    `M_red * 1_free`, which under-counts contributions from columns
    corresponding to constrained DOFs, breaking partition of unity near
    Dirichlet boundaries.

The row-sum lumped mass is the correct quantity for explicit central
difference dynamics (`a = M^{-1} f`) and for mass-diagonal Jacobi
preconditioning, because it preserves partition of unity (total mass =
density × volume) by construction.
"""
function assemble_lumped_mass!(
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
  return_type = AssembledVector()
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
Return the assembled lumped mass vector after a call to
`assemble_lumped_mass!`.  Extracts the free-DOF subset (non-condensed
mode) or returns the full vector (condensed mode).

Note: shares backing storage with `residual(asm)`.  Callers that need
both quantities must `copy` this result before any subsequent
`assemble_vector!` / `assemble_lumped_mass!` / `assemble_diagonal!` call.
"""
function lumped_mass(asm::AbstractAssembler)
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
