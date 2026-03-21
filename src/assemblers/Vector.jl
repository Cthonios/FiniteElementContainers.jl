"""
$(TYPEDSIGNATURES)
"""
function assemble_vector!(
  assembler, func::F, Uu, p
) where F <: Function
  assemble_vector!(
    assembler.residual_storage, 
    assembler.vector_pattern, assembler.dof,
    func, Uu, p
  )
  return nothing
end

"""
$(TYPEDSIGNATURES)
"""
function assemble_vector!(
  storage, pattern, dof, func::F, Uu, p
) where F <: Function
  fill!(storage, zero(eltype(storage)))
  backend = KA.get_backend(storage)
  fspace = function_space(dof)
  X = coordinates(p)
  t = current_time(p)
  Δt = time_step(p)
  U = p.field
  U_old = p.field_old
  _update_for_assembly!(p, dof, Uu)
  return_type = AssembledVector()
  conns = fspace.elem_conns
  for (b, (
    block_physics, ref_fe, props
  )) in enumerate(zip(
    values(p.physics), values(fspace.ref_fes),
    values(p.properties)
  ))
    _assemble_block!(
      backend,
      storage,
      conns.data, conns.nelems[b], conns.offsets[b], 
      pattern.block_start_indices[b], pattern.block_el_level_sizes[b],
      func,
      block_physics, ref_fe,
      X, t, Δt,
      U, U_old,
      block_view(p.state_old, b), block_view(p.state_new, b), props,
      return_type
    )
  end
  
  return nothing
end
