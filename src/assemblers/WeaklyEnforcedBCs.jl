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
    assembler.residual_storage,
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
    assembler.residual_storage,
    p.field, coordinates(p), p.neumann_bcs
  )
  return nothing
end

# below method implicitly will not zero out arrays
"""
$(TYPEDSIGNATURES)
"""
function assemble_vector_weakly_enforced_bc!(
  storage, U, X, bcs
)
  # do not zero!
  for bc in values(bcs.bc_caches)
    _assemble_block_vector_weakly_enforced_bc!(
      storage,
      U, X,
      bc.element_conns.data, bc.ref_fe, bc.sides, bc.vals
    )
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
    conn = connectivity(ref_fe, conns, e, 1)
    x_el = _element_level_fields(X, ref_fe, conn)#s, E)
    R_el = _element_scratch_vector(boundary_element(ref_fe, side), U)
  
    for q in 1:num_surface_quadrature_points(ref_fe)
      # interps = _surface_interpolants(ref_fe, q, side)
      interps = MappedH1OrL2SurfaceInterpolants(ref_fe, x_el, q, side)
      Nvec = interps.N_reduced
      JxW = interps.JxW
  
      f_val = vals[q, e]
      R_el = R_el + JxW * reduce(vcat, ntuple(i -> Nvec[i] * f_val, length(Nvec)))
    end
  
    surf_conns = surface_connectivity(ref_fe, conns, side, e, 1)
    _assemble_element!(field, R_el, surf_conns, e, 0, 0)
  end
end
