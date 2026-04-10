"""
$(TYPEDSIGNATURES)
"""
function assemble_vector!(
  assembler, func::F, Uu, p
) where F <: Function
  assemble_vector!(
    assembler.residual_storage, 
    assembler.vector_pattern, assembler.dof,
    func, Uu, p;
    use_inplace_methods = _use_inplace_methods(assembler),
    use_sparse_vector = _use_sparse_vector(assembler),
  )
  return nothing
end

"""
$(TYPEDSIGNATURES)
"""
function assemble_vector!(
  storage, pattern, dof, func::F, Uu, p;
  use_inplace_methods::Bool = false,
  use_sparse_vector::Bool = false
) where F <: Function
  fill!(storage, zero(eltype(storage)))
  fspace = function_space(dof)
  X = coordinates(p)
  t = current_time(p)
  Δt = time_step(p)
  U = p.field
  U_old = p.field_old
  _update_for_assembly!(p, dof, Uu)
  return_type = AssembledVector()
  conns = fspace.elem_conns
  foreach_block(conns, p.physics, p.properties, fspace.ref_fes) do physics, props, ref_fe, b
    if use_sparse_vector
      field = block_view(storage, pattern, b)
    else
      field = storage
    end

    if use_inplace_methods
      _assemble_block!(
        field,
        func,
        physics,
        t, Δt,
        props,
        block_view(p.state_old, b), block_view(p.state_new, b),
        conns.data, conns.offsets[b], ref_fe, X, U, U_old
      )
    else
      _assemble_block!(
        field,
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
  
  return nothing
end
