function assemble_mass!(
  assembler, func::F, Uu, p
) where F <: Function
  assemble_matrix!(
    assembler.mass_storage, assembler.matrix_pattern, assembler.dof,
    func, Uu, p
  )
end

function assemble_stiffness!(
  assembler, func::F, Uu, p
) where F <: Function
  assemble_matrix!(
    assembler.stiffness_storage, assembler.matrix_pattern, assembler.dof,
    func, Uu, p
  )
end

"""
$(TYPEDSIGNATURES)
Note this is hard coded to storing the assembled sparse matrix in 
the stiffness_storage field of assembler.
"""
function assemble_matrix!(
  storage, pattern, dof, func::F, Uu, p
) where F <: Function
  fill!(storage, zero(eltype(storage)))
  backend = KA.get_backend(storage)
  fspace = function_space(dof)
  t = current_time(p.times)
  dt = time_step(p.times)
  _update_for_assembly!(p, dof, Uu)
  return_type = AssembledMatrix()
  conns = fspace.elem_conns.data
  for (
    nelem, coffset,
    block_start_index, block_el_level_size,
    block_physics, ref_fe, 
    state_old, state_new, props
  ) in zip(
    values(fspace.elem_conns.nelems),
    values(fspace.elem_conns.offsets), 
    values(pattern.block_start_indices),
    values(pattern.block_el_level_sizes),
    values(p.physics), values(fspace.ref_fes),
    values(p.state_old), values(p.state_new),
    values(p.properties),
  )
    _assemble_block!(
      backend,
      storage,
      conns, nelem, coffset,       
      block_start_index, block_el_level_size,
      func,
      block_physics, ref_fe,
      p.h1_coords, t, dt,
      p.h1_field, p.h1_field_old, 
      state_old, state_new, props,
      return_type
    )
  end
end
