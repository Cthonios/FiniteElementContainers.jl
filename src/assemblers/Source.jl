# Assembly of body force contributions to the residual vector.
#
# Computes -∫ Nᵢ · b dΩ for each element block where body forces are defined.
# Does NOT zero the residual storage — adds to the existing assembled residual
# (same convention as NeumannBC assembly).

"""
$(TYPEDSIGNATURES)
Assemble body force contributions into the residual vector.
"""
function assemble_vector_source!(assembler, Uu, p)
  assemble_vector_source!(
    assembler.residual_storage,
    assembler.vector_pattern, assembler.dof,
    Uu, p
  )
  return nothing
end

function assemble_vector_source!(storage, pattern, dof, Uu, p)
  isempty(p.sources.source_caches) && return nothing

  _update_for_assembly!(p, dof, Uu)
  fspace = function_space(dof)
  U = p.field
  X = coordinates(p)
  sources = p.sources
  conns = fspace.elem_conns
  for (block_id, block_name, source) in zip(
    sources.source_block_ids, sources.source_block_names,
    values(sources.source_caches)
  )
    ref_fe = getfield(fspace.ref_fes, block_name)
    vals   = source.vals

    _assemble_block_vector_source!(
      storage,
      conns.data, conns.offsets[block_id],
      ref_fe, X, U, vals,
    )
  end
  return nothing
end

function _assemble_block_vector_source!(
  field, conns, coffset, ref_fe, X, U, vals
)
  fec_foraxes(vals, 2) do e
    conn = connectivity(ref_fe, conns, e, coffset)
    X_el = _element_level_fields_flat(X, ref_fe, conn)
    R_el = _element_scratch(AssembledVector(), ref_fe, U)
  
    for q in 1:num_cell_quadrature_points(ref_fe)
      interps = _cell_interpolants(ref_fe, q)
      interps = map_interpolants(interps, X_el)
      Nvec = interps.N
      JxW  = interps.JxW
      b_val = vals[q, e]
  
      R_el = R_el - JxW * reduce(vcat, ntuple(i -> Nvec[i] * b_val, length(Nvec)))
    end
  
    @views _assemble_element!(field, R_el, conn, e, 0, 0)
  end
end
