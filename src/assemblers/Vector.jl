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
  t = current_time(p.times)
  Δt = time_step(p.times)
  _update_for_assembly!(p, dof, Uu)
  return_type = AssembledVector()
  for (
    conns,
    block_physics, ref_fe,
    state_old, state_new, props
  ) in zip(
    values(fspace.elem_conns), 
    values(p.physics), values(fspace.ref_fes),
    values(p.state_old), values(p.state_new),
    values(p.properties)
  )
    _assemble_block!(
      backend,
      storage,
      conns, 0, 0, # NOTE these are never used for vectors
      # TODO eventually we'll need them if we want to use sparse vectors
      func,
      block_physics, ref_fe,
      p.h1_coords, t, Δt,
      p.h1_field, p.h1_field_old,
      state_old, state_new, props,
      return_type
    )
  end
  
  return nothing
end
