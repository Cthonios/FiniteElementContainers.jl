function assemble_mass!(
  assembler, func::F, Uu, p,
  enzyme_safe::Bool = false
) where F <: Function
  assemble_matrix!(
    assembler.mass_storage, assembler.matrix_pattern, assembler.dof,
    func, Uu, p,
    enzyme_safe
  )
end

function assemble_stiffness!(
  assembler, func::F, Uu, p,
  enzyme_safe::Bool = false
) where F <: Function
  assemble_matrix!(
    assembler.stiffness_storage, assembler.matrix_pattern, assembler.dof,
    func, Uu, p,
    enzyme_safe
  )
end

"""
$(TYPEDSIGNATURES)
Note this is hard coded to storing the assembled sparse matrix in 
the stiffness_storage field of assembler.
"""
function assemble_matrix!(
  storage, pattern, dof, func::F, Uu, p,
  enzyme_safe::Bool = false
) where F <: Function
  fill!(storage, zero(eltype(storage)))
  fspace = function_space(dof)
  X = coordinates(p)
  t = current_time(p)
  dt = time_step(p)
  U = p.field
  U_old = p.field_old
  _update_for_assembly!(p, dof, Uu, enzyme_safe)
  return_type = AssembledMatrix()
  conns = fspace.elem_conns
  for (b, (
    block_physics, ref_fe, props
  )) in enumerate(zip(
    values(p.physics), values(fspace.ref_fes),
    values(p.properties),
  ))
    _assemble_block!(
      # backend,
      storage,
      # conns.data, conns.nelems[b], conns.offsets[b],
      conns.data, conns.offsets[b],
      pattern.block_start_indices[b], pattern.block_el_level_sizes[b],
      func,
      block_physics, ref_fe,
      X, t, dt,
      U, U_old, 
      block_view(p.state_old, b), block_view(p.state_new, b), props,
      return_type,
      enzyme_safe
    )
  end
end
