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
      0, 0, # NOTE these are never used for vectors
      # TODO eventually we'll need them if we want to use sparse vectors
      func,
      block_physics, ref_fe,
      p.h1_coords, t, Δt,
      p.h1_field, p.h1_field_old,
      block_view(p.state_old, b), block_view(p.state_new, b), props,
      return_type
    )
  end
  
  return nothing
end
