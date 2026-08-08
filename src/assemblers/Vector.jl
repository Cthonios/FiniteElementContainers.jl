"""
$(TYPEDSIGNATURES)
"""
function assemble_vector!(
  assembler::AbstractAssembler, func::F, Uu, p
) where F <: Function
  if _use_sparse_vector(assembler)
    storage = assembler.residual_unknowns
  else
    storage = assembler.residual_storage
  end
  assemble_vector!(
    storage,
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
  foreach_block(fspace, p) do physics, ref_fe, b
    # if use_sparse_vector
    #   field = block_view(storage, pattern, b)
    # else
    #   field = storage
    # end
    field = storage

    if use_inplace_methods
      _assemble_block!(
        field,
        func,
        b,
        physics,
        t, Δt,
        p.properties, p.state_old, p.state_new,
        conns,
        ref_fe, X, U, U_old
      )
    else
      _assemble_block!(
        field,
        conns,
        func,
        b,
        physics, ref_fe,
        X, t, Δt,
        U, U_old,
        p.state_old, p.state_new, p.properties,
        return_type
      )
    end
  end
  
  return nothing
end
