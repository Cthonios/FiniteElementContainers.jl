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
  fspace = function_space(dof)
  X = coordinates(p)
  t = current_time(p)
  dt = time_step(p)
  U = p.field
  U_old = p.field_old
  _update_for_assembly!(p, dof, Uu)
  return_type = AssembledMatrix()
  conns = fspace.elem_conns
  foreach_block(fspace, p) do physics, props, ref_fe, b
    _assemble_block!(
      storage,
      conns.data, conns.offsets[b],
      pattern.block_start_indices[b], pattern.block_el_level_sizes[b],
      func,
      physics, ref_fe,
      X, t, dt,
      U, U_old, 
      block_view(p.state_old, b), block_view(p.state_new, b), props,
      return_type
    )
  end
end
