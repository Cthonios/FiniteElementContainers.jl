"""
$(TYPEDSIGNATURES)
"""
function assemble_scalar!(
  assembler, func::F, Uu, p
) where F <: Function
  # the nothing below is because
  # there is no needed sparsity pattern
  assemble_quadrature_quantity!(
    assembler.scalar_quadrature_storage, 
    nothing, assembler.dof,
    func, Uu, p
  )
end

"""
$(TYPEDSIGNATURES)
"""
function assemble_quadrature_quantity!(
  storage::L2Field, pattern, dof,
  func::F, Uu, p,
  return_type::AssembledReturnType = AssembledScalar()
) where F <: Function
  fspace = function_space(dof)
  X = coordinates(p)
  t = current_time(p)
  Δt = time_step(p)
  U = p.field
  U_old = p.field_old
  _update_for_assembly!(p, dof, Uu)
  conns = fspace.elem_conns
  foreach_block(fspace, p) do physics, props, ref_fe, b
    _assemble_block!(
      block_view(storage, b),
      conns.data, conns.offsets[b], 
      func,
      physics, ref_fe,
      X, t, Δt,
      U, U_old,
      block_view(p.state_old, b), block_view(p.state_new, b), props,
      return_type
    )
  end
end

"""
$(TYPEDSIGNATURES)
"""
function assemble_quadrature_quantity!(
  storage::NamedTuple, pattern, dof,
  func::F, Uu, p,
  return_type::AssembledReturnType = AssembledScalar()
) where F <: Function
  fspace = function_space(dof)
  X = coordinates(p)
  t = current_time(p)
  Δt = time_step(p)
  U = p.field
  U_old = p.field_old
  _update_for_assembly!(p, dof, Uu)
  conns = fspace.elem_conns
  for (b, (
    block_storage,
    block_physics, ref_fe, props
  )) in enumerate(zip(
    values(storage),
    values(p.physics), values(fspace.ref_fes),
    values(p.properties)
  ))
    _assemble_block!(
      # backend,
      block_storage,
      conns.data, conns.offsets[b], 
      func,
      block_physics, ref_fe,
      X, t, Δt,
      U, U_old,
      block_view(p.state_old, b), block_view(p.state_new, b), props,
      return_type
    )
  end
end
