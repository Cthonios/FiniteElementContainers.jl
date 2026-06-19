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
  # _update_for_assembly!(p, assembler.dof, Uu)
  update_bc_values!(p.robin_bcs, assembler, coordinates(p), current_time(p), p.field)
  assemble_vector_weakly_enforced_bc!(
    assembler.residual_storage, assembler.dof,
    p.field, coordinates(p), p.robin_bcs
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

# really only needed for stiffness contribution of robin bc
# need the atomic here in the edge case that two sides shared by an
# element are accessed, e.g. a corner element in a sideset
function assemble_matrix_robin_bc!(assembler, Uu, p)
  # _check_matrix_assembly_supported(assembler, "assemble_matrix_robin_bc!")
  # _update_for_assembly!(p, assembler.dof, Uu)
  update_bc_values!(p.robin_bcs, assembler, coordinates(p), current_time(p), p.field)
  _assemble_matrix_weakly_enforced_bc!(
    assembler.stiffness_storage, assembler.matrix_pattern, assembler.dof,
    coordinates(p), p.robin_bcs
  )
  return nothing
end

function _assemble_matrix_weakly_enforced_bc!(storage, pattern, dof, X, bcs)
  fspace = function_space(dof)
  foreach_block(fspace) do ref_fe, b
    block_id = bcs.block_id_to_bc[b]
    block_id == -1 && return
    cache = bcs.bc_caches[block_id]
    _assemble_block_matrix_weakly_enforced_bc!(
      block_view(storage, pattern, b), dof, X,
      cache.element_conns.data, ref_fe, cache.elements, cache.sides, cache.dvalsdu
    )
  end
  return nothing
end

function _assemble_block_matrix_weakly_enforced_bc!(
  storage, dof, X::AbstractField,
  conns, ref_fe, elements, sides, dvalsdu
)
  ND   = size(dof, 1)
  NEPE = ReferenceFiniteElements.num_cell_dofs(ref_fe)

  fec_foreach(sides) do e
    side  = sides[e]
    el_id = elements[e]   # local-to-block index — matches SparseMatrixPattern's element loop
    conn  = connectivity(ref_fe, conns, e, 1)
    surf_conns = surface_connectivity(ref_fe, conns, side, e, 1)

    raw_ids = indexin(surf_conns, conn)
    @assert all(!isnothing, raw_ids) "Robin side nodes must be a subset of the parent element's nodes"
    node_to_face_idx = ntuple(NEPE) do n
      pos = findfirst(==(n), raw_ids)
      pos === nothing ? 0 : pos
    end

    x_el = _element_level_fields(X, ref_fe, conn)
    K_el = zeros(SMatrix{ND * NEPE, ND * NEPE, Float64, (ND * NEPE)^2})

    for q in 1:num_surface_quadrature_points(ref_fe)
      interps = MappedH1OrL2SurfaceInterpolants(ref_fe, x_el, q, side)
      Nvec = interps.N_reduced
      JxW  = interps.JxW
      dval = dvalsdu[q, e]   # SMatrix{ND, ND}
      K_el = K_el + _expand_face_block(Nvec, JxW, dval, node_to_face_idx, Val(NEPE), Val(ND))
    end

    _assemble_element_add!(storage, K_el, -1, conn, el_id)
  end
  return nothing
end

function _assemble_element_add!(
  storage, K_el::SMatrix{NDOF1, NDOF2, T, NDOF1xNDOF2},
  ::Int,  # conns unused, kept for signature symmetry with _assemble_element!
  conns,
  el_id::Int
) where {NDOF1, NDOF2, T, NDOF1xNDOF2}
  start_id = (el_id - 1) * NDOF1xNDOF2 + 1
  for (i, id) in enumerate(start_id:start_id+NDOF1xNDOF2-1)
    # fec_atomic_add!(storage, id, K_el.data[i])
    Atomix.@atomic storage[id] += K_el.data[i]
  end
  return nothing
end

@inline function _expand_face_block(
  Nvec, JxW, dval::SMatrix{ND, ND}, node_to_face_idx, ::Val{NEPE}, ::Val{ND}
) where {ND, NEPE}
  NxNDof = ND * NEPE
  return SMatrix{NxNDof, NxNDof}(
    ntuple(NxNDof * NxNDof) do lin
      row = (lin - 1) % NxNDof + 1
      col = (lin - 1) ÷ NxNDof + 1
      ni, di = (row - 1) ÷ ND + 1, (row - 1) % ND + 1
      nj, dj = (col - 1) ÷ ND + 1, (col - 1) % ND + 1
      ii, jj = node_to_face_idx[ni], node_to_face_idx[nj]
      (ii == 0 || jj == 0) ? 0.0 : JxW * Nvec[ii] * Nvec[jj] * dval[di, dj]
    end
  )
end