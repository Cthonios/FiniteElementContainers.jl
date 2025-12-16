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
  storage, pattern, dof,
  func::F, Uu, p, return_type = AssembledScalar()
) where F <: Function
  backend = KA.get_backend(p.h1_field)
  fspace = function_space(dof)
  t = current_time(p.times)
  Δt = time_step(p.times)
  _update_for_assembly!(p, dof, Uu)
  # return_type = AssembledScalar()
  for (
    block_storage,
    conns, 
    block_physics, ref_fe,
    state_old, state_new, props
  ) in zip(
    values(storage),
    values(fspace.elem_conns), 
    values(p.physics), values(fspace.ref_fes),
    values(p.state_old), values(p.state_new),
    values(p.properties)
  )
    _assemble_block!(
      backend,
      block_storage,
      conns, 0, 0,
      func,
      block_physics, ref_fe,
      p.h1_coords, t, Δt,
      p.h1_field, p.h1_field_old,
      state_old, state_new, props,
      return_type
    )
  end
end
