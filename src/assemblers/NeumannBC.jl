"""
$(TYPEDSIGNATURES)
"""
function assemble_vector_neumann_bc!(
  assembler, Uu, p
)
  assemble_vector_neumann_bc!(
    assembler.residual_storage, 
    assembler.dof,
    Uu, p
  )
  return nothing
end

# # below method implicitly will not zero out arrays
"""
$(TYPEDSIGNATURES)
"""
function assemble_vector_neumann_bc!(
  storage, dof, Uu, p
)
  # do not zero!
  # TODO should below 2 methods calls be assumed to have
  # been conducted previously?
  backend = KA.get_backend(p.h1_field)
  _update_for_assembly!(p, dof, Uu)
  for bc in values(p.neumann_bcs.bc_caches)
    _assemble_block_vector_neumann_bc!(
      backend, storage,
      p.h1_field, p.h1_coords,
      bc.element_conns.data, bc.ref_fe, bc.sides, bc.vals,
      bc.element_conns.nelems[1] 
    )
  end
end

"""
$(TYPEDSIGNATURES)
"""
function _assemble_block_vector_neumann_bc!(
  ::KA.CPU,
  field::AbstractField, U::AbstractField, X::AbstractField,
  conns, ref_fe, sides, vals, nelem
)
  for e in 1:nelem
    conn = connectivity(ref_fe, conns, e, 1)
    x_el = _element_level_fields(X, ref_fe, conn)#s, e)
    R_el = _element_scratch_vector(surface_element(ref_fe.element), U)
    side = sides[e]
    for q in 1:num_quadrature_points(surface_element(ref_fe.element))
      interps = MappedSurfaceInterpolants(ref_fe, x_el, q, side)
      Nvec = interps.N_reduced
      JxW = interps.JxW

      # TODO Clean this up
      f_val = vals[q, e]
      if length(f_val) == 1
        f_val = f_val[1]
      end

      R_el = R_el + JxW * Nvec * f_val
    end
    surf_conns = surface_connectivity(ref_fe, conns, e, 1)
    @views _assemble_element!(field, R_el, surf_conns, e, 0, 0)
  end
end

"""
$(TYPEDSIGNATURES)
Kernel for residual block assembly of neumann bcs

TODO mark const fields
"""
# COV_EXCL_START
KA.@kernel function _assemble_block_vector_neumann_bc_kernel!(
  field::AbstractField, U::AbstractField, X::AbstractField,
  conns, ref_fe, sides, vals
)
  E = KA.@index(Global)

  conn = connectivity(ref_fe, conns, E, 1)
  x_el = _element_level_fields(X, ref_fe, conn)#s, E)
  R_el = _element_scratch_vector(surface_element(ref_fe.element), U)
  side = sides[E]

  for q in 1:num_quadrature_points(surface_element(ref_fe.element))
    interps = MappedSurfaceInterpolants(ref_fe, x_el, q, side)
    Nvec = interps.N_reduced
    JxW = interps.JxW

    # TODO Clean this up
    f_val = vals[q, E]
    if length(f_val) == 1
      f_val = f_val[1]
    end

    R_el = R_el + JxW * Nvec * f_val
  end

  surf_conns = surface_connectivity(ref_fe, conns, E, 1)
  _assemble_element!(field, R_el, surf_conns, E, 0, 0)
end
# COV_EXCL_STOP

"""
$(TYPEDSIGNATURES)
"""
function _assemble_block_vector_neumann_bc!(
  backend::KA.Backend,
  field::AbstractField, 
  U::AbstractField, X::AbstractField,
  conns, ref_fe, sides, vals, nelem
)
  kernel! = _assemble_block_vector_neumann_bc_kernel!(backend)
  kernel!(
    field, U, X,
    conns, ref_fe, sides, vals,
    ndrange = nelem
  )
  return nothing
end
