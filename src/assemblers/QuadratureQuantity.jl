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
      backend,
      block_storage,
      conns.data, conns.nelems[b], conns.offsets[b], 
      0, 0,
      func,
      block_physics, ref_fe,
      p.h1_coords, t, Δt,
      p.h1_field, p.h1_field_old,
      # state_old, state_new, props,
      block_view(p.state_old, b), block_view(p.state_new, b), props,
      return_type
    )
  end
end
