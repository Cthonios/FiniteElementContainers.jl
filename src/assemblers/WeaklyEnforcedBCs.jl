"""
$(TYPEDSIGNATURES)
"""
function assemble_vector_neumann_bc!(
  assembler, Uu, p
)
  # TODO should below method call be assumed to have
  # been conducted previously?
  _update_for_assembly!(p, assembler.dof, Uu)
  assemble_vector_weakly_enforced_bc!(
    assembler.residual_storage, assembler.dof,
    p.field, coordinates(p), p.neumann_bcs
  )
  return nothing
end

"""
$(TYPEDSIGNATURES)
"""
function assemble_vector_robin_bc!(
  assembler, Uu, p
)
  # TODO should below method call be assumed to have
  # been conducted previously?
  _update_for_assembly!(p, assembler.dof, Uu)
  update_bc_values!(p.robin_bcs, coordinates(p), current_time(p), p.field)
  assemble_vector_weakly_enforced_bc!(
    assembler.residual_storage, assembler.dof,
    p.field, coordinates(p), p.neumann_bcs
  )
  return nothing
end

# below method implicitly will not zero out arrays
"""
$(TYPEDSIGNATURES)
"""
function assemble_vector_weakly_enforced_bc!(
  storage, dof, U, X, bcs
)
  fspace = function_space(dof)
  foreach_block(fspace) do ref_fe, b
    block_id = bcs.block_id_to_bc[b]
    if block_id == -1
      # do nothing
    else
      cache = bcs.bc_caches[block_id]
      _assemble_block_vector_weakly_enforced_bc!(
        storage,
        U, X,
        cache.element_conns.data, ref_fe, cache.sides, cache.vals
      )
    end
  end
end

"""
$(TYPEDSIGNATURES)
TODO this method needs to be updated to allow for
sparse vectors
TODO this method is hardcoded to H1 or L2 spaces
"""
function _assemble_block_vector_weakly_enforced_bc!(
  field::AbstractField, 
  U::AbstractField, X::AbstractField,
  conns, ref_fe, sides, vals
)
  fec_foreach(sides) do e
    side = sides[e]
    conn = connectivity(ref_fe, conns, e, 1) # 1 for coffset
    surf_conns = surface_connectivity(ref_fe, conns, side, e, 1) # 1 for coffset
    x_el = _element_level_fields(X, ref_fe, conn)
  
    for q in 1:num_surface_quadrature_points(ref_fe)
      interps = MappedH1OrL2SurfaceInterpolants(ref_fe, x_el, q, side)
      Nvec = interps.N_reduced
      JxW = interps.JxW
  
      f_val = vals[q, e]
      form = GeneralFormulation{num_fields(X), num_fields(field)}()
      # NOTE e is obviously not correct below but it isn't used
      # in this method anyway when field is an AbstractField
      scatter_with_values!(field, form, e, surf_conns, Nvec, JxW * f_val)
    end
  end
end
